/*
 * rockchip_drv_video.c — VA-API driver for Rockchip RK3588 via MPP
 *
 * Copyright (C) 2026 Eduardo García-Mádico Portabella <woodyst@gmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#define _GNU_SOURCE
#include <va/va_backend.h>
#include <va/va_drmcommon.h>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <sys/ioctl.h>
#include <errno.h>

#include "h264.h"

/* ── logging ─────────────────────────────────────────────────── */
static FILE *g_log_fp = NULL;
static void log_init(void) {
    const char *p = getenv("RK_VAAPI_LOG");
    if (p && *p) g_log_fp = fopen(p, "a");
}
#define LOG(fmt, ...) do { if (g_log_fp) \
    fprintf(g_log_fp, "[rk-vaapi pid=%d] " fmt "\n", getpid(), ##__VA_ARGS__); \
} while(0)

/* ── limits ──────────────────────────────────────────────────── */
#define MAX_CONFIGS   16
#define MAX_CONTEXTS   8
#define MAX_SURFACES  64
#define MAX_BUFFERS  256

/* VA object ID namespaces */
#define CONFIG_ID_BASE   0x01000000u
#define CONTEXT_ID_BASE  0x02000000u
#define SURFACE_ID_BASE  0x03000000u
#define BUFFER_ID_BASE   0x04000000u

/* ── data structures ─────────────────────────────────────────── */

typedef struct {
    bool          used;
    VAProfile     profile;
    VAEntrypoint  entrypoint;
} RKConfig;

/* One complete access unit handed to the decode worker.  `buf` is malloc'd;
 * the worker owns it, feeds it to MPP and frees it. */
#define RK_JOB_CAP 12
typedef struct {
    uint8_t *data;      /* complete Annex-B access unit */
    size_t   len;
    VASurfaceID sid;    /* routed back via mpp packet pts */
    int      nosync;    /* no surface waits on this job (VP9 altref) */
} RKJob;

typedef struct {
    bool         used;
    VAConfigID   config_id;
    int          width, height;

    MppCtx       mpp;
    MppApi      *mpi;
    MppCodingType coding;

    /* External decode buffer group (DRM dma-heap, falling back to ION).
     * Mirrors ffmpeg rkmppdec.c: MPP allocates its DPB / output buffers from
     * this group so 4K decode does NOT eat the small kernel CMA pool
     * (256 MB default), which otherwise exhausts mid-clip. */
    MppBufferGroup ext_group;

    /* buffers collected between BeginPicture / EndPicture */
    VABufferID   pending[64];
    int          n_pending;

    VASurfaceID  render_target;

    /* decode worker: MPP put/get is ONLY touched by this thread once the
     * context starts it.  VA-API entry points enqueue access units here and
     * wait on surface condvars instead of polling MPP. */
    pthread_t         dec_thread;
    pthread_mutex_t   jq_mtx;
    pthread_cond_t    jq_not_empty;   /* worker waits (bounded timeout) */
    pthread_cond_t    jq_not_full;    /* producer waits for a free slot */
    RKJob             jq[RK_JOB_CAP];
    int               jq_head, jq_tail, jq_n;
    volatile int      dec_stop;
    void             *dec_d;          /* RKDriver * (forward type) */

    /* Surface IDs handed to MPP, in submission (decode) order.  MPP emits
     * output frames in the same order, so when a frame for sid X arrives,
     * every queued entry before X that never arrived was dropped by the
     * hardware — resolve those surfaces instantly (keep previous content)
     * instead of letting SyncSurface sit out its 350ms timeout per drop. */
    VASurfaceID       awaited[64];
    int               awa_head, awa_tail;

    /* H.264 state for SPS/PPS reconstruction */
    VAPictureParameterBufferH264 last_pp;
    bool         sps_sent;

    /* decode health: frames actually delivered and when the last one was,
     * used to distinguish an occasional MPP frame drop (tolerate, keep the
     * previous content) from a fully stalled decoder (report an error). */
    RK_S64        frames_out;
    struct timespec last_out_ts;
} RKContext;

typedef struct {
    bool         used;
    int          width, height;

    /* filled after decode */
    MppFrame     frame;          /* always NULL (kept for ABI compat) */
    int          prime_fd;       /* dup'd fd to priv_buf, stable for surface lifetime */
    int          hstride;
    int          vstride;
    bool         priv_filled;    /* priv_buf/last_frame ever held a decoded frame */

    /* Zero-copy export (M3): the most recent decoded MppFrame is kept alive on
     * the surface until the surface is reused (BeginPicture).  Its dma-buf fd
     * is exported by vaExportSurfaceHandle with no per-frame memcpy; the
     * buffer returns to MPP's pool on reuse (the consumer has presented the
     * frame by then). */
    MppFrame     last_frame;
    int          export_fd;      /* dup'd dma-buf of last_frame, -1 if none */

    /* Dedicated per-surface DMA-BUF used as permanent output buffer.
     * Decoded pixels are copied here in assign_mpp_frame so MPP can
     * immediately reuse its internal 3-buffer pool, preventing the
     * buffer aliasing that causes wrong frames in Firefox's compositor.
     * Also serves as the pre-decode placeholder for ExportSurfaceHandle
     * capability probes (Firefox DMABUF probe before any decode). */
    MppBufferGroup priv_group;
    MppBuffer      priv_buf;

    MppFrameFormat fmt;     /* pixel format of last decoded frame (0 = NV12 default) */
    bool         decoded;
    VAContextID  ctx_id;   /* context currently decoding into this surface */
    pthread_mutex_t  lock;
    pthread_cond_t   cond;
} RKSurface;

typedef struct {
    bool           used;
    VABufferType   type;
    unsigned int   size;
    unsigned int   num_elements;
    void          *data;
} RKBuffer;

typedef struct {
    RKConfig   configs [MAX_CONFIGS];
    RKContext  contexts[MAX_CONTEXTS];
    RKSurface  surfaces[MAX_SURFACES];
    RKBuffer   buffers [MAX_BUFFERS];

    /* Serializes live object allocation / destruction.  ffmpeg decodes on
     * one thread while the output thread vaCreateImage()/vaDestroyImage()s,
     * both calling rk_CreateBuffer(); without this lock they can hand out
     * the same slot and one thread's vaDestroyBuffer() frees the other
     * thread's image buffer (=> vaGetImage "invalid surface"). */
    pthread_mutex_t obj_mtx;
} RKDriver;

/* ── helpers ─────────────────────────────────────────────────── */

static RKDriver *drv_from_ctx(VADriverContextP ctx) {
    return (RKDriver *)ctx->pDriverData;
}

static RKConfig *config_by_id(RKDriver *d, VAConfigID id) {
    unsigned idx = id - CONFIG_ID_BASE;
    return (idx < MAX_CONFIGS && d->configs[idx].used) ? &d->configs[idx] : NULL;
}

static RKContext *context_by_id(RKDriver *d, VAContextID id) {
    unsigned idx = id - CONTEXT_ID_BASE;
    return (idx < MAX_CONTEXTS && d->contexts[idx].used) ? &d->contexts[idx] : NULL;
}

static RKSurface *surface_by_id(RKDriver *d, VASurfaceID id) {
    unsigned idx = id - SURFACE_ID_BASE;
    return (idx < MAX_SURFACES && d->surfaces[idx].used) ? &d->surfaces[idx] : NULL;
}

static RKBuffer *buffer_by_id(RKDriver *d, VABufferID id) {
    unsigned idx = id - BUFFER_ID_BASE;
    return (idx < MAX_BUFFERS && d->buffers[idx].used) ? &d->buffers[idx] : NULL;
}

static MppCodingType profile_to_coding(VAProfile p) {
    switch (p) {
    case VAProfileH264ConstrainedBaseline:
    case VAProfileH264Main:
    case VAProfileH264High:
#if VA_MINOR_VERSION >= 18
    case VAProfileH264High10:
#endif
                                 return MPP_VIDEO_CodingAVC;
    case VAProfileHEVCMain:
    case VAProfileHEVCMain10:    return MPP_VIDEO_CodingHEVC;
    case VAProfileVP8Version0_3: return MPP_VIDEO_CodingVP8;
    case VAProfileVP9Profile0:
    case VAProfileVP9Profile2:   return MPP_VIDEO_CodingVP9;
    case VAProfileAV1Profile0:
    case VAProfileAV1Profile1:   return MPP_VIDEO_CodingAV1;
    default:                     return MPP_VIDEO_CodingUnused;
    }
}

static int profile_idc(VAProfile p) {
    switch (p) {
    case VAProfileH264ConstrainedBaseline: return 66;
    case VAProfileH264Main:                return 77;
    case VAProfileH264High:                return 100;
#if VA_MINOR_VERSION >= 18
    case VAProfileH264High10:              return 110;
#endif
    default:                               return 100;
    }
}

/* forward decls for the decode worker (defined further down) */
static void rk_ctx_stop_worker(RKContext *c);
static void *rk_decode_thread(void *arg);
static VAStatus rk_enqueue_job(RKContext *c, const uint8_t *data, size_t len,
                               VASurfaceID sid, int nosync);
static void assign_mpp_frame(MppFrame frame, RKContext *c, RKDriver *d);

/* ── VADriverVTable implementations ──────────────────────────── */

static VAStatus rk_Terminate(VADriverContextP ctx) {
    LOG("Terminate: cleaning up driver");
    RKDriver *d = drv_from_ctx(ctx);
    if (!d) return VA_STATUS_SUCCESS;

    /* Stop all decode workers BEFORE freeing surfaces/contexts: a worker may
     * still touch surfaces while draining MPP output. */
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (!d->contexts[i].used) continue;
        rk_ctx_stop_worker(&d->contexts[i]);
    }

    /* destroy any leftover objects */
    for (int i = 0; i < MAX_SURFACES; i++) {
        if (!d->surfaces[i].used) continue;
        if (d->surfaces[i].frame) mpp_frame_deinit(&d->surfaces[i].frame);
        if (d->surfaces[i].last_frame) mpp_frame_deinit(&d->surfaces[i].last_frame);
        if (d->surfaces[i].export_fd > 0) close(d->surfaces[i].export_fd);
        if (d->surfaces[i].prime_fd >= 0) close(d->surfaces[i].prime_fd);
        if (d->surfaces[i].priv_buf)   mpp_buffer_put(d->surfaces[i].priv_buf);
        if (d->surfaces[i].priv_group) mpp_buffer_group_put(d->surfaces[i].priv_group);
        pthread_cond_destroy(&d->surfaces[i].cond);
        pthread_mutex_destroy(&d->surfaces[i].lock);
    }
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (!d->contexts[i].used) continue;
        if (d->contexts[i].mpp) {
            d->contexts[i].mpi->reset(d->contexts[i].mpp);
            mpp_destroy(d->contexts[i].mpp);
        }
        if (d->contexts[i].ext_group)
            mpp_buffer_group_put(d->contexts[i].ext_group);
        pthread_mutex_destroy(&d->contexts[i].jq_mtx);
        pthread_cond_destroy(&d->contexts[i].jq_not_empty);
        pthread_cond_destroy(&d->contexts[i].jq_not_full);
    }
    for (int i = 0; i < MAX_BUFFERS; i++) {
        if (d->buffers[i].used) free(d->buffers[i].data);
    }
    pthread_mutex_destroy(&d->obj_mtx);
    free(d);
    ctx->pDriverData = NULL;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_QueryConfigProfiles(VADriverContextP ctx,
                                       VAProfile *list, int *n) {
    (void)ctx;
    int i = 0;
    list[i++] = VAProfileH264ConstrainedBaseline;
    list[i++] = VAProfileH264Main;
    list[i++] = VAProfileH264High;
#if VA_MINOR_VERSION >= 18
    list[i++] = VAProfileH264High10;
#endif
    list[i++] = VAProfileHEVCMain;
    list[i++] = VAProfileHEVCMain10;
    list[i++] = VAProfileVP8Version0_3;
    list[i++] = VAProfileVP9Profile0;
    list[i++] = VAProfileVP9Profile2;
    /* AV1 not advertised: MPP needs a full OBU bytestream but VA-API hands us
     * only headerless tile data, so MPP can never parse it. Firefox falls back
     * to VP9 (hardware-decoded) for AV1-capable content. */
    *n = i;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_QueryConfigEntrypoints(VADriverContextP ctx,
                                          VAProfile profile,
                                          VAEntrypoint *list, int *n) {
    (void)ctx;
    if (profile_to_coding(profile) == MPP_VIDEO_CodingUnused)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    list[0] = VAEntrypointVLD;
    *n = 1;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_GetConfigAttributes(VADriverContextP ctx,
                                       VAProfile profile,
                                       VAEntrypoint entrypoint,
                                       VAConfigAttrib *list, int n) {
    (void)ctx; (void)profile; (void)entrypoint;
    for (int i = 0; i < n; i++) {
        LOG("GetConfigAttributes: type=%d", list[i].type);
        switch (list[i].type) {
        case VAConfigAttribRTFormat:
            list[i].value = VA_RT_FORMAT_YUV420 | VA_RT_FORMAT_YUV420_10;
            break;
        case VAConfigAttribDecSliceMode:
            list[i].value = VA_DEC_SLICE_MODE_NORMAL;
            break;
        case VAConfigAttribMaxPictureWidth:
            list[i].value = 8192;
            break;
        case VAConfigAttribMaxPictureHeight:
            list[i].value = 8192;
            break;
        case VAConfigAttribEncryption:
            list[i].value = VA_ATTRIB_NOT_SUPPORTED;
            break;
        default:
            LOG("GetConfigAttributes: unsupported type=%d", list[i].type);
            list[i].value = VA_ATTRIB_NOT_SUPPORTED;
        }
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_CreateConfig(VADriverContextP ctx,
                                 VAProfile profile, VAEntrypoint entrypoint,
                                 VAConfigAttrib *attribs, int n_attribs,
                                 VAConfigID *out_id) {
    RKDriver *d = drv_from_ctx(ctx);
    LOG("CreateConfig: profile=%d entrypoint=%d n_attribs=%d",
        profile, entrypoint, n_attribs);

    if (profile_to_coding(profile) == MPP_VIDEO_CodingUnused) {
        LOG("CreateConfig: unsupported profile %d", profile);
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    }
    if (entrypoint != VAEntrypointVLD) {
        LOG("CreateConfig: unsupported entrypoint %d", entrypoint);
        return VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT;
    }
    (void)attribs; (void)n_attribs;

    for (unsigned i = 0; i < MAX_CONFIGS; i++) {
        pthread_mutex_lock(&d->obj_mtx);
        if (!d->configs[i].used) {
            d->configs[i].used = true;
            d->configs[i].profile = profile;
            d->configs[i].entrypoint = entrypoint;
            *out_id = CONFIG_ID_BASE + i;
            pthread_mutex_unlock(&d->obj_mtx);
            return VA_STATUS_SUCCESS;
        }
        pthread_mutex_unlock(&d->obj_mtx);
    }
    return VA_STATUS_ERROR_ALLOCATION_FAILED;
}

static VAStatus rk_DestroyConfig(VADriverContextP ctx, VAConfigID id) {
    RKDriver *d = drv_from_ctx(ctx);
    pthread_mutex_lock(&d->obj_mtx);
    RKConfig *c = config_by_id(d, id);
    if (!c) { pthread_mutex_unlock(&d->obj_mtx); return VA_STATUS_ERROR_INVALID_CONFIG; }
    c->used = false;
    pthread_mutex_unlock(&d->obj_mtx);
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_QueryConfigAttributes(VADriverContextP ctx,
                                          VAConfigID id,
                                          VAProfile *profile,
                                          VAEntrypoint *entrypoint,
                                          VAConfigAttrib *attribs, int *n) {
    RKDriver *d = drv_from_ctx(ctx);
    RKConfig *c = config_by_id(d, id);
    if (!c) return VA_STATUS_ERROR_INVALID_CONFIG;
    *profile = c->profile;
    *entrypoint = c->entrypoint;
    *n = 0;
    /* Chromium's FillProfileInfo (vaapi_wrapper.cc) considers a profile
     * unusable unless vaQueryConfigAttributes returns a non-empty RTFormat,
     * so report the formats this driver actually decodes into. */
    if (attribs) {
        attribs[0].type  = VAConfigAttribRTFormat;
        attribs[0].value = VA_RT_FORMAT_YUV420 | VA_RT_FORMAT_YUV420_10;
        *n = 1;
    }
    return VA_STATUS_SUCCESS;
}

/* vaCreateSurfaces (old API, redirected) */
static VAStatus rk_CreateSurfaces(VADriverContextP ctx,
                                   int width, int height, int format,
                                   int n, VASurfaceID *ids) {
    RKDriver *d = drv_from_ctx(ctx);
    (void)format;

    int allocated = 0;
    for (int s = 0; s < n; s++) {
        unsigned i;
        for (i = 0; i < MAX_SURFACES; i++) {
            if (!d->surfaces[i].used) break;
        }
        if (i == MAX_SURFACES) {
            /* roll back — must free placeholder buffers before zeroing */
            for (int j = 0; j < allocated; j++) {
                unsigned idx = ids[j] - SURFACE_ID_BASE;
                RKSurface *rb = &d->surfaces[idx];
                if (rb->prime_fd >= 0) close(rb->prime_fd);
                if (rb->priv_buf)   mpp_buffer_put(rb->priv_buf);
                if (rb->priv_group) mpp_buffer_group_put(rb->priv_group);
                pthread_mutex_destroy(&rb->lock);
                pthread_cond_destroy(&rb->cond);
                memset(rb, 0, sizeof(RKSurface));
            }
            return VA_STATUS_ERROR_ALLOCATION_FAILED;
        }
        RKSurface *surf = &d->surfaces[i];
        memset(surf, 0, sizeof(*surf));
        surf->used       = true;
        surf->width      = width;
        surf->height     = height;
        surf->prime_fd   = -1;
        surf->export_fd  = -1;   /* fd 0 is a valid-return but we treat <=0 as "none" */
        surf->last_frame = NULL;

        /* Pre-allocate placeholder DMA-BUF so ExportSurfaceHandle succeeds
         * before any decode (e.g. Firefox's DMABUF capability probe). */
        {
            unsigned hs = (unsigned)((width  + 15) & ~15);
            unsigned vs = (unsigned)((height + 15) & ~15);
            MppBufferGroup grp = NULL;
            MppBuffer      buf = NULL;
            if (mpp_buffer_group_get_internal(&grp, MPP_BUFFER_TYPE_DRM) == MPP_OK &&
                mpp_buffer_get(grp, &buf, hs * vs * 3) == MPP_OK) {
                int raw_fd = mpp_buffer_get_fd(buf);
                int dup_fd = (raw_fd > 0) ? dup(raw_fd) : -1;
                if (dup_fd > 0) {
                    /* Neutral placeholder (gray NV12, NOT zeroed = green) so
                     * any pre-decode/stale export never shows a green frame. */
                    uint8_t *ph = (uint8_t *)mpp_buffer_get_ptr(buf);
                    if (ph) {
                        memset(ph, 80, (size_t)hs * vs);
                        memset(ph + (size_t)hs * vs, 128,
                               (size_t)hs * vs * 3 - (size_t)hs * vs);
                    }
                    surf->priv_group = grp;
                    surf->priv_buf   = buf;
                    surf->prime_fd   = dup_fd;
                    surf->hstride    = (int)hs;
                    surf->vstride    = (int)vs;
                    LOG("CreateSurfaces: surface %ux%u placeholder prime_fd=%d",
                        (unsigned)width, (unsigned)height, surf->prime_fd);
                } else {
                    LOG("CreateSurfaces: mpp_buffer_get_fd failed (raw_fd=%d), no placeholder", raw_fd);
                    mpp_buffer_put(buf);
                    mpp_buffer_group_put(grp);
                }
            } else {
                if (buf) mpp_buffer_put(buf);
                if (grp) mpp_buffer_group_put(grp);
                LOG("CreateSurfaces: placeholder alloc failed, prime_fd=-1");
            }
        }

        pthread_mutex_init(&surf->lock, NULL);
        pthread_cond_init(&surf->cond, NULL);
        ids[s] = SURFACE_ID_BASE + i;
        allocated++;
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_DestroySurfaces(VADriverContextP ctx,
                                    VASurfaceID *list, int n) {
    LOG("DestroySurfaces: n=%d", n);
    RKDriver *d = drv_from_ctx(ctx);
    for (int i = 0; i < n; i++) {
        RKSurface *s = surface_by_id(d, list[i]);
        if (!s) continue;
        if (s->frame)      mpp_frame_deinit(&s->frame);
        if (s->last_frame) mpp_frame_deinit(&s->last_frame);
        if (s->export_fd > 0) close(s->export_fd);
        if (s->prime_fd >= 0) close(s->prime_fd);
        if (s->priv_buf)   { mpp_buffer_put(s->priv_buf);        s->priv_buf   = NULL; }
        if (s->priv_group) { mpp_buffer_group_put(s->priv_group); s->priv_group = NULL; }
        pthread_cond_destroy(&s->cond);
        pthread_mutex_destroy(&s->lock);
        memset(s, 0, sizeof(*s));
    }
    return VA_STATUS_SUCCESS;
}

/* vaCreateSurfaces2 (new API with attributes) */
static VAStatus rk_CreateSurfaces2(VADriverContextP ctx,
                                    unsigned int format,
                                    unsigned int width, unsigned int height,
                                    VASurfaceID *ids, unsigned int n,
                                    VASurfaceAttrib *attribs,
                                    unsigned int n_attribs) {
    LOG("CreateSurfaces2: %ux%u fmt=0x%x n=%u n_attribs=%u",
        width, height, format, n, n_attribs);
    for (unsigned i = 0; i < n_attribs; i++) {
        LOG("  attrib[%u] type=%d flags=%d value=0x%x",
            i, attribs[i].type, attribs[i].flags,
            attribs[i].value.type == VAGenericValueTypeInteger
                ? (unsigned)attribs[i].value.value.i : 0u);
    }
    return rk_CreateSurfaces(ctx, (int)width, (int)height, (int)format,
                              (int)n, ids);
}

static VAStatus rk_CreateContext(VADriverContextP ctx,
                                  VAConfigID config_id,
                                  int width, int height,
                                  int flag,
                                  VASurfaceID *targets, int n_targets,
                                  VAContextID *out_id) {
    RKDriver *d = drv_from_ctx(ctx);
    (void)flag; (void)targets; (void)n_targets;

    RKConfig *cfg = config_by_id(d, config_id);
    if (!cfg) return VA_STATUS_ERROR_INVALID_CONFIG;

    MppCodingType coding = profile_to_coding(cfg->profile);
    if (coding == MPP_VIDEO_CodingUnused)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;

    for (unsigned i = 0; i < MAX_CONTEXTS; i++) {
        if (d->contexts[i].used) continue;
        RKContext *c = &d->contexts[i];
        memset(c, 0, sizeof(*c));

        LOG("CreateContext: config=0x%x %dx%d coding=%d",
            config_id, width, height, (int)coding);

        MPP_RET ret = mpp_create(&c->mpp, &c->mpi);
        if (ret != MPP_OK) {
            LOG("mpp_create FAILED: %d", ret);
            return VA_STATUS_ERROR_ALLOCATION_FAILED;
        }
        LOG("CreateContext: mpp_create OK");

        ret = mpp_init(c->mpp, MPP_CTX_DEC, coding);
        if (ret != MPP_OK) {
            mpp_destroy(c->mpp);
            LOG("mpp_init FAILED: %d (coding=%d)", ret, (int)coding);
            return VA_STATUS_ERROR_ALLOCATION_FAILED;
        }
        LOG("CreateContext: mpp_init OK");

        /* Must be set after mpp_init: split_parse=0 means we send complete
         * access units; OUTPUT_BLOCK=0 makes decode_get_frame non-blocking. */
        MppDecCfg dec_cfg = NULL;
        mpp_dec_cfg_init(&dec_cfg);
        mpp_dec_cfg_set_u32(dec_cfg, "base:split_parse", 0);
        c->mpi->control(c->mpp, MPP_DEC_SET_CFG, dec_cfg);
        mpp_dec_cfg_deinit(dec_cfg);

        /* Emit frames as soon as they are decoded (rkmpp option).  Without it
         * MPP holds finished pictures in its reorder buffer waiting for more
         * input; a VA-API consumer that syncs one surface right after each
         * EndPicture then waits out the 3s timeout at the end of a clip. */
        MppParam imm_out = (MppParam)(intptr_t)1;
        c->mpi->control(c->mpp, MPP_DEC_SET_IMMEDIATE_OUT, (MppParam)&imm_out);

        /* rkmpp parity knobs (harmless with the async worker; earlier they
         * were measured on the broken 1-2fps engine and moved nothing). */
        c->mpi->control(c->mpp, MPP_DEC_SET_PARSER_FAST_MODE, (MppParam)&imm_out);
        c->mpi->control(c->mpp, MPP_DEC_SET_DISABLE_ERROR,   (MppParam)&imm_out);

        int block = 0;
        c->mpi->control(c->mpp, MPP_SET_OUTPUT_BLOCK, (MppParam)&block);

        /* Non-blocking input so the worker never blocks inside MPP when we
         * ask it to stop (decode_put_packet may return MPP_ERR_BUFFER_FULL,
         * which the worker retries after draining). */
        c->mpi->control(c->mpp, MPP_SET_INPUT_BLOCK, (MppParam)&block);

        /* Route MPP's decode buffers (DPB + output) out of CMA into system
         * dma-heap memory via an external group, like ffmpeg's rkmppdec. */
        c->ext_group = NULL;
        if (mpp_buffer_group_get_internal(&c->ext_group, MPP_BUFFER_TYPE_DRM) != MPP_OK)
            mpp_buffer_group_get_internal(&c->ext_group, MPP_BUFFER_TYPE_ION);
        if (c->ext_group) {
            /* Raise the pool so zero-copy pinning (last_frame kept until the
             * surface is reused) + MPP's DPB coexist. */
            RK_S32 max_frames = 48;
            mpp_buffer_group_limit_config(c->ext_group, 0, max_frames);
            c->mpi->control(c->mpp, MPP_DEC_SET_EXT_BUF_GROUP, (MppParam)c->ext_group);
            LOG("CreateContext: EXT_BUF_GROUP allocated (%p)", (void *)c->ext_group);
        } else {
            LOG("CreateContext: EXT_BUF_GROUP allocation FAILED (4K may fail)");
        }

        /* decode worker pump */
        c->dec_d   = d;
        c->jq_head = c->jq_tail = c->jq_n = 0;
        c->dec_stop = 0;
        pthread_mutex_init(&c->jq_mtx, NULL);
        pthread_cond_init(&c->jq_not_empty, NULL);
        pthread_cond_init(&c->jq_not_full, NULL);
        if (pthread_create(&c->dec_thread, NULL, rk_decode_thread, c) != 0) {
            LOG("CreateContext: decode worker thread FAILED");
            pthread_mutex_destroy(&c->jq_mtx);
            pthread_cond_destroy(&c->jq_not_empty);
            pthread_cond_destroy(&c->jq_not_full);
            mpp_destroy(c->mpp);
            memset(c, 0, sizeof(*c));
            return VA_STATUS_ERROR_ALLOCATION_FAILED;
        }
        LOG("CreateContext: decode worker thread started");

        c->used      = true;
        c->config_id = config_id;
        c->width     = width;
        c->height    = height;
        c->coding    = coding;
        c->sps_sent  = false;

        *out_id = CONTEXT_ID_BASE + i;
        return VA_STATUS_SUCCESS;
    }
    return VA_STATUS_ERROR_ALLOCATION_FAILED;
}

static VAStatus rk_DestroyContext(VADriverContextP ctx, VAContextID id) {
    LOG("DestroyContext: ctx=0x%x", id);
    RKDriver *d = drv_from_ctx(ctx);
    RKContext *c = context_by_id(d, id);
    if (!c) return VA_STATUS_ERROR_INVALID_CONTEXT;
    rk_ctx_stop_worker(c);
    /* Flush the decoder before destroying it (mirrors ffmpeg rkmppdec:
     * mpi->reset() waits for in-flight hardware tasks / frees the DPB so
     * mpp_destroy() does not race a still-running decode task and crash
     * inside MPP's deinit with a NULL mutex). */
    if (c->mpp) {
        c->mpi->reset(c->mpp);
        mpp_destroy(c->mpp);
    }
    if (c->ext_group) { mpp_buffer_group_put(c->ext_group); c->ext_group = NULL; }
    pthread_mutex_destroy(&c->jq_mtx);
    pthread_cond_destroy(&c->jq_not_empty);
    pthread_cond_destroy(&c->jq_not_full);
    memset(c, 0, sizeof(*c));
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_CreateBuffer(VADriverContextP ctx,
                                 VAContextID context,
                                 VABufferType type,
                                 unsigned int size,
                                 unsigned int num_elements,
                                 void *data,
                                 VABufferID *out_id) {
    RKDriver *d = drv_from_ctx(ctx);
    (void)context;

    pthread_mutex_lock(&d->obj_mtx);
    for (unsigned i = 0; i < MAX_BUFFERS; i++) {
        if (d->buffers[i].used) continue;
        RKBuffer *b = &d->buffers[i];
        b->used         = true;
        b->type         = type;
        b->size         = size;
        b->num_elements = num_elements;
        b->data         = malloc((size_t)size * num_elements);
        if (!b->data) {
            b->used = false;
            pthread_mutex_unlock(&d->obj_mtx);
            return VA_STATUS_ERROR_ALLOCATION_FAILED;
        }
        if (data) memcpy(b->data, data, (size_t)size * num_elements);
        else      memset(b->data, 0,    (size_t)size * num_elements);
        *out_id = BUFFER_ID_BASE + i;
        pthread_mutex_unlock(&d->obj_mtx);
        return VA_STATUS_SUCCESS;
    }
    pthread_mutex_unlock(&d->obj_mtx);
    return VA_STATUS_ERROR_ALLOCATION_FAILED;
}

static VAStatus rk_BufferSetNumElements(VADriverContextP ctx,
                                         VABufferID id, unsigned int n) {
    RKDriver *d = drv_from_ctx(ctx);
    RKBuffer *b = buffer_by_id(d, id);
    if (!b) return VA_STATUS_ERROR_INVALID_BUFFER;
    b->num_elements = n;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_MapBuffer(VADriverContextP ctx,
                              VABufferID id, void **ptr) {
    RKDriver *d = drv_from_ctx(ctx);
    RKBuffer *b = buffer_by_id(d, id);
    if (!b) return VA_STATUS_ERROR_INVALID_BUFFER;
    *ptr = b->data;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_UnmapBuffer(VADriverContextP ctx, VABufferID id) {
    (void)ctx; (void)id;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_DestroyBuffer(VADriverContextP ctx, VABufferID id) {
    RKDriver *d = drv_from_ctx(ctx);
    pthread_mutex_lock(&d->obj_mtx);
    RKBuffer *b = buffer_by_id(d, id);
    if (!b) { pthread_mutex_unlock(&d->obj_mtx); return VA_STATUS_ERROR_INVALID_BUFFER; }
    free(b->data);
    b->data = NULL;
    b->used = false;
    pthread_mutex_unlock(&d->obj_mtx);
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_BeginPicture(VADriverContextP ctx,
                                 VAContextID ctx_id,
                                 VASurfaceID render_target) {
    LOG("BeginPicture: ctx=0x%x surface=0x%x", ctx_id, render_target);
    RKDriver  *d = drv_from_ctx(ctx);
    RKContext *c = context_by_id(d, ctx_id);
    if (!c) return VA_STATUS_ERROR_INVALID_CONTEXT;

    c->render_target = render_target;
    c->n_pending     = 0;

    /* Reset surface state for this decode cycle.  priv_buf/prime_fd are kept
     * intact so ExportSurfaceHandle always returns a valid fd (the previous
     * decoded frame or the initial placeholder).  The zero-copy pin (last_frame)
     * is released here — the consumer has presented the previous frame before
     * reusing this surface, so its buffer can return to MPP's pool. */
    RKSurface *s = surface_by_id(d, render_target);
    if (s) {
        pthread_mutex_lock(&s->lock);
        if (s->last_frame) { mpp_frame_deinit(&s->last_frame); s->last_frame = NULL; }
        if (s->export_fd > 0) { close(s->export_fd); s->export_fd = -1; }
        s->decoded = false;
        s->ctx_id  = ctx_id;
        pthread_mutex_unlock(&s->lock);
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_RenderPicture(VADriverContextP ctx,
                                  VAContextID ctx_id,
                                  VABufferID *buffers, int n) {
    RKDriver  *d = drv_from_ctx(ctx);
    RKContext *c = context_by_id(d, ctx_id);
    if (!c) return VA_STATUS_ERROR_INVALID_CONTEXT;

    for (int i = 0; i < n && c->n_pending < 64; i++) {
        c->pending[c->n_pending++] = buffers[i];
        /* Snapshot VAPictureParameterBufferH264 immediately */
        RKBuffer *b = buffer_by_id(d, buffers[i]);
        if (b && b->type == VAPictureParameterBufferType &&
            c->coding == MPP_VIDEO_CodingAVC) {
            memcpy(&c->last_pp, b->data,
                   sizeof(VAPictureParameterBufferH264));
        }
    }
    return VA_STATUS_SUCCESS;
}

/* Route one MPP output frame to the right surface and mark it decoded.
 * Shared by EndPicture poll loops and the SyncSurface drain loop. */
static void assign_mpp_frame(MppFrame frame, RKContext *c, RKDriver *d)
{
    if (mpp_frame_get_info_change(frame)) {
        LOG("assign_mpp_frame: info_change → acknowledged, render_target=0x%x",
            (unsigned)c->render_target);
        c->mpi->control(c->mpp, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
        mpp_frame_deinit(&frame);
        return;
    }

    RK_S64      raw_pts = mpp_frame_get_pts(frame);
    VASurfaceID sid     = (VASurfaceID)raw_pts;
    RKSurface  *s       = sid ? surface_by_id(d, sid) : NULL;

    /* Pure PTS routing: every access unit carries its target VASurfaceID as
     * the packet pts, so a frame can only ever be delivered to the surface it
     * belongs to.  No `render_target` fallback here — that mutable per-context
     * field was the cause of SyncSurface 3s timeouts under mpv's multi-surface
     * pipeline (frame for surface A arriving while render_target had moved on). */
    if (!s) {
        LOG("assign_mpp_frame: PTS=0x%llx no matching surface, dropped",
            (unsigned long long)raw_pts);
        mpp_frame_deinit(&frame);
        return;
    }

    /* MPP emits frames in submission (decode) order.  A frame arriving for
     * sid proves every older awaited sid that never came was dropped by the
     * hardware — resolve those surfaces right now (keep the previous content)
     * so VA-API SyncSurface does not sit out a 350ms timeout per dropped
     * frame (the visible micro-stutter at 4K). */
    while (c->awa_head != c->awa_tail) {
        VASurfaceID p = c->awaited[c->awa_head % 64];
        c->awa_head++;
        if (p == sid) break;
        RKSurface *sp = surface_by_id(d, p);
        if (sp) {
            pthread_mutex_lock(&sp->lock);
            sp->decoded = true;
            pthread_cond_signal(&sp->cond);
            pthread_mutex_unlock(&sp->lock);
            LOG("worker: DROPPED_R=0x%x stale (next ok 0x%x)", (unsigned)p, (unsigned)sid);
        }
    }

    MppBuffer      buf    = mpp_frame_get_buffer(frame);
    int            fwidth = (int)mpp_frame_get_width(frame);
    int            fheight= (int)mpp_frame_get_height(frame);
    int            fhs    = (int)mpp_frame_get_hor_stride(frame);
    int            fvs    = (int)mpp_frame_get_ver_stride(frame);
    MppFrameFormat ffmt   = mpp_frame_get_fmt(frame);

    /* Detect MPP decode-failure frames (blank/zero content = one-frame green
     * flash at 4K).  Sample a 3x3 grid of Y pixels (edges + center) so a
     * genuine frame with ANY content is never misjudged (top-left corners are
     * often letterbox-black). */
    RK_U32 errp = mpp_frame_get_errinfo(frame);
    RK_U32 disc = mpp_frame_get_discard(frame);
    bool blank = false;
    if (buf && fwidth > 0 && fheight > 0) {
        const uint8_t *ypg = (const uint8_t *)mpp_buffer_get_ptr(buf);
        if (ypg) {
            const int pr[3] = { 1, fheight / 2, fheight - 2 };
            const int pc[3] = { 8, fwidth  / 2, fwidth  - 9 };
            RK_S64 acc = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    acc += ypg[(size_t)(pr[r] * (fhs > 0 ? fhs : fwidth)) + (size_t)pc[c]];
            blank = (acc == 0);
        }
    }
    LOG("assign: sid=0x%x err=%u disc=%u blank=%d",
        (unsigned)sid, (unsigned)errp, (unsigned)disc, blank ? 1 : 0);

    /* MPP decode-failure protection: a blank/zeroed or error-flagged frame
     * would display as a one-frame green flash.  Treat it as a dropped frame
     * and KEEP the previous content (or neutral gray) instead of copying
     * garbage. */
    if (errp || blank) {
        LOG("assign: sid=0x%x MPP blank/err frame dropped (err=%u disc=%u blank=%d)",
            (unsigned)sid, (unsigned)errp, (unsigned)disc, blank ? 1 : 0);
        pthread_mutex_lock(&s->lock);
        s->decoded = true;
        pthread_cond_signal(&s->cond);
        pthread_mutex_unlock(&s->lock);
        mpp_frame_deinit(&frame);
        return;
    }

    int  copy_w = fwidth  > 0 ? fwidth  : s->width;
    int  copy_h = fheight > 0 ? fheight : s->height;
    int  src_hs = fhs > 0 ? fhs : copy_w;
    int  src_vs = fvs > 0 ? fvs : copy_h;
    bool i10    = MPP_FRAME_FMT_IS_YUV_10BIT(ffmt);
    int  bpp    = i10 ? 2 : 1;
    int  copied = 0;

    /* Default = DPB->priv_buf copy path (aliasing-free, stable per-surface
     * buffer => no tearing for consumers that hold textures across reuse,
     * e.g. mpv).  RK_VAAPI_ZEROCOPY=1 switches to the M3 zero-copy export
     * (pin last MppFrame + export its dma-buf) for benchmarking — faster
     * but MPP may recycle the buffer while GL is still drawing it. */
    static int keep_copy = -1;
    if (keep_copy < 0) {
        const char *zc = getenv("RK_VAAPI_ZEROCOPY");
        keep_copy = (zc && zc[0] == '1') ? 0 : 1;
    }

    if (keep_copy) {
        void *src = buf ? mpp_buffer_get_ptr(buf) : NULL;
        void *dst = s->priv_buf ? mpp_buffer_get_ptr(s->priv_buf) : NULL;
        if (src && dst) {
            const uint8_t *sy = (const uint8_t *)src;
            uint8_t       *dy = (uint8_t       *)dst;
            for (int r = 0; r < copy_h; r++)
                memcpy(dy + (size_t)r * src_hs * bpp,
                       sy + (size_t)r * src_hs * bpp,
                       (size_t)src_hs * bpp);
            const uint8_t *su = sy + (size_t)src_hs * src_vs * bpp;
            uint8_t       *du = dy + (size_t)src_hs * src_vs * bpp;
            for (int r = 0; r < copy_h / 2; r++)
                memcpy(du + (size_t)r * src_hs * bpp,
                       su + (size_t)r * src_hs * bpp,
                       (size_t)src_hs * bpp);
            copied = 1;
        }
    }

    /* Replace the previously pinned frame (buffer returns to MPP pool). */
    pthread_mutex_lock(&s->lock);
    if (s->last_frame) { mpp_frame_deinit(&s->last_frame); s->last_frame = NULL; }
    if (s->export_fd > 0) { close(s->export_fd); s->export_fd = -1; }
    if (!keep_copy) {
        s->last_frame = frame;      /* keep; deinit on surface reuse */
        int mfd = buf ? mpp_buffer_get_fd(buf) : -1;
        if (mfd > 0) {
            int dfd = dup(mfd);
            s->export_fd = (dfd > 0) ? dfd : -1;
        } else {
            s->export_fd = -1;
        }
    } else {
        mpp_frame_deinit(&frame);   /* pixels already copied to priv_buf */
    }
    s->frame  = NULL;
    s->fmt    = ffmt;
    s->priv_filled = true;   /* this surface now holds a real decoded frame */
    if (fwidth  > 0) s->width   = fwidth;
    if (fheight > 0) s->height  = fheight;
    if (fhs     > 0) s->hstride = fhs;
    if (fvs     > 0) s->vstride = fvs;
    s->decoded  = true;
    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->lock);
    /* touch these outside the lock; only the worker thread writes them */
    c->frames_out++;
    clock_gettime(CLOCK_MONOTONIC, &c->last_out_ts);
    LOG("assign_mpp_frame: surface=0x%x prime_fd=%d export_fd=%d MPP %dx%d stride=%dx%d fmt=0x%x copied=%d",
        (unsigned)sid, s->prime_fd, s->export_fd, fwidth, fheight, fhs, fvs, (unsigned)ffmt, copied);
}

/* ── decode worker (single-threaded MPP consumer) ──────────────── */

/* Stop the worker and free any yet-unconsumed access units.  Never called
 * while the worker could still dereference `c` (join strictly before any
 * context storage is reused). */
static void rk_ctx_stop_worker(RKContext *c)
{
    if (!c->dec_thread) return;
    c->dec_stop = 1;
    pthread_cond_signal(&c->jq_not_empty);
    pthread_join(c->dec_thread, NULL);
    c->dec_thread = 0;

    pthread_mutex_lock(&c->jq_mtx);
    while (c->jq_n > 0) {
        RKJob *j = &c->jq[c->jq_head];
        free(j->data);
        j->data = NULL;
        c->jq_head = (c->jq_head + 1) % RK_JOB_CAP;
        c->jq_n--;
    }
    pthread_mutex_unlock(&c->jq_mtx);
}

/* Worker thread: the ONLY thread that calls decode_put_packet /
 * decode_get_frame for this context.  VA-API entry points (EndPicture,
 * SyncSurface) previously polled decode_get_frame from multiple threads,
 * which made MPP return/route frames wrong and waited out the full 3s
 * SyncSurface drain ~40% of the time at 4K.  Now they enqueue AUs and wait
 * on surface condition variables instead. */
static void *rk_decode_thread(void *arg)
{
    RKContext *c = (RKContext *)arg;
    RKDriver  *d = (RKDriver *)c->dec_d;

    while (!c->dec_stop) {
        /* 1) harvest whatever MPP produced so far (non-blocking get) */
        MppFrame f = NULL;
        while (c->mpi->decode_get_frame(c->mpp, &f) == MPP_OK && f) {
            assign_mpp_frame(f, c, d);
            f = NULL;
        }

        /* 2) wait for a job (100ms timeout keeps draining late MPP output) */
        pthread_mutex_lock(&c->jq_mtx);
        while (c->jq_n == 0 && !c->dec_stop) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 100 * 1000 * 1000;
            if (ts.tv_nsec >= 1000000000) { ts.tv_sec += 1; ts.tv_nsec -= 1000000000; }
            if (pthread_cond_timedwait(&c->jq_not_empty, &c->jq_mtx, &ts) == ETIMEDOUT)
                break;
        }
        if (c->dec_stop) {
            pthread_mutex_unlock(&c->jq_mtx);
            break;
        }
        int have = c->jq_n > 0;
        RKJob job;
        if (have) {
            job = c->jq[c->jq_head];        /* struct copy (data is a ptr) */
            c->jq_head = (c->jq_head + 1) % RK_JOB_CAP;
            c->jq_n--;
        }
        pthread_cond_signal(&c->jq_not_full);
        pthread_mutex_unlock(&c->jq_mtx);

        if (!have) continue;    /* idle timeout; loop drains again */

        /* 3) feed MPP.  decode_put_packet can legitimately return
         *    MPP_ERR_BUFFER_FULL (-1012 in this MPP) when the hardware is
         *    still chewing on the previous access units (easy at 4K: a DPB of
         *    frames in flight fills the input queue).  Never drop the AU —
         *    pumping output in between gives the parser room, and the retry
         *    only stops when the context is being torn down. */
        if (job.len) {
            MPP_RET ret = MPP_OK;
            while (!c->dec_stop) {
                MppPacket pkt = NULL;
                mpp_packet_init(&pkt, job.data, job.len);
                mpp_packet_set_length(pkt, job.len);
                mpp_packet_set_pts(pkt, (RK_S64)job.sid);
                ret = c->mpi->decode_put_packet(c->mpp, pkt);
                mpp_packet_deinit(&pkt);
                if (ret != MPP_ERR_BUFFER_FULL)
                    break;
                /* drain one output to make room, then retry shortly */
                MppFrame f2 = NULL;
                if (c->mpi->decode_get_frame(c->mpp, &f2) == MPP_OK && f2) {
                    assign_mpp_frame(f2, c, d);
                    continue;
                }
                struct timespec ts;
                ts.tv_sec  = 0;
                ts.tv_nsec = 2000 * 1000;   /* 2ms */
                nanosleep(&ts, NULL);
            }
            if (!c->dec_stop) {
                if (ret != MPP_OK)
                    LOG("decode worker: put_packet ret=%d len=%zu sid=0x%x",
                        (int)ret, job.len, (unsigned)job.sid);
                else {
                    c->awaited[c->awa_tail % 64] = job.sid;
                    c->awa_tail++;
                    LOG("decode worker: put len=%zu sid=0x%x", job.len, (unsigned)job.sid);
                }
            }
        }
        free(job.data);
    }
    return NULL;
}

/* Copy one access unit into the job ring for the worker.  Blocks (bounded
 * only by context teardown) when the ring is full: natural backpressure —
 * EndPicture must never fail "queue full" just because MPP is briefly busy,
 * or ffmpeg aborts the stream on a spurious error. */
static VAStatus rk_enqueue_job(RKContext *c, const uint8_t *data, size_t len,
                               VASurfaceID sid, int nosync)
{
    if (c->dec_stop || !data || !len) return VA_STATUS_SUCCESS;

    pthread_mutex_lock(&c->jq_mtx);
    while (c->jq_n >= RK_JOB_CAP && !c->dec_stop) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 1000 * 1000;          /* 1ms */
        if (ts.tv_nsec >= 1000000000) { ts.tv_sec += 1; ts.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&c->jq_not_full, &c->jq_mtx, &ts);
    }
    if (c->dec_stop) {
        pthread_mutex_unlock(&c->jq_mtx);
        return VA_STATUS_SUCCESS;           /* tearing down; drop silently */
    }
    void *copy = malloc(len);
    if (!copy) {
        pthread_mutex_unlock(&c->jq_mtx);
        return VA_STATUS_ERROR_ALLOCATION_FAILED;
    }
    memcpy(copy, data, len);
    RKJob *j = &c->jq[c->jq_tail];
    j->data   = copy;
    j->len    = len;
    j->sid    = sid;
    j->nosync = nosync;
    c->jq_tail = (c->jq_tail + 1) % RK_JOB_CAP;
    c->jq_n++;
    pthread_cond_signal(&c->jq_not_empty);
    pthread_mutex_unlock(&c->jq_mtx);
    return VA_STATUS_SUCCESS;
}

/* Build Annex B bitstream from VA-API buffers and send to MPP */
static VAStatus do_h264_decode(RKContext *c, RKDriver *d)
{
    /* gather slice data */
    uint8_t *pkt_data = NULL;
    size_t   pkt_cap  = 0;
    size_t   pkt_sz   = 0;

#define PKT_APPEND(ptr, len) do {                           \
    size_t _l = (len);                                      \
    if (pkt_sz + _l > pkt_cap) {                           \
        pkt_cap = (pkt_sz + _l) * 2 + 4096;               \
        pkt_data = realloc(pkt_data, pkt_cap);             \
        if (!pkt_data) return VA_STATUS_ERROR_ALLOCATION_FAILED; \
    }                                                       \
    memcpy(pkt_data + pkt_sz, (ptr), _l);                 \
    pkt_sz += _l;                                          \
} while (0)

    /* check if first slice is IDR to prepend SPS+PPS */
    bool is_idr = false;
    for (int i = 0; i < c->n_pending; i++) {
        RKBuffer *b = buffer_by_id(d, c->pending[i]);
        if (!b || b->type != VASliceDataBufferType) continue;
        uint8_t nal_type = ((uint8_t *)b->data)[0] & 0x1F;
        is_idr = (nal_type == 5);
        break;
    }

    /* SPS + PPS before every IDR (or first frame) */
    if (is_idr || !c->sps_sent) {
        uint8_t hdr[512];
        int n = h264_write_sps(hdr, sizeof(hdr), &c->last_pp,
                               profile_idc(config_by_id(d, c->config_id)->profile));
        if (n > 0) { PKT_APPEND(hdr, (size_t)n); }

        n = h264_write_pps(hdr, sizeof(hdr), &c->last_pp);
        if (n > 0) { PKT_APPEND(hdr, (size_t)n); }

        c->sps_sent = true;
    }

    /* append each slice with Annex B start code */
    static const uint8_t sc[4] = {0x00, 0x00, 0x00, 0x01};
    for (int i = 0; i < c->n_pending; i++) {
        RKBuffer *b = buffer_by_id(d, c->pending[i]);
        if (!b || b->type != VASliceDataBufferType) continue;
        PKT_APPEND(sc, 4);
        PKT_APPEND(b->data, (size_t)b->size * b->num_elements);
    }
#undef PKT_APPEND

    if (!pkt_sz) {
        LOG("do_h264_decode: no slice data, marking surface 0x%x decoded",
            (unsigned)c->render_target);
        free(pkt_data);
        RKSurface *tgt = surface_by_id(d, c->render_target);
        if (tgt) {
            pthread_mutex_lock(&tgt->lock);
            tgt->decoded = true;
            pthread_cond_signal(&tgt->cond);
            pthread_mutex_unlock(&tgt->lock);
        }
        return VA_STATUS_SUCCESS;
    }

    /* Hand the full access unit to the decode worker.  The worker owns
     * decode_put_packet / decode_get_frame; VA-API threads never touch MPP
     * directly (that concurrent access previously stalled SyncSurface). */
    LOG("do_h264_decode: submit %zu bytes target=0x%x",
        pkt_sz, (unsigned)c->render_target);
    VAStatus st = rk_enqueue_job(c, pkt_data, pkt_sz, c->render_target, 0);
    free(pkt_data);
    return st;
}

/* Return false for VP9 altref / non-displayed frames (show_frame=0).
 * These are decoded by MPP internally as references but never output via
 * decode_get_frame — polling for them stalls the thread for the full timeout.
 * VP9 uncompressed header layout (MSB-first):
 *   [7:6] frame_marker=0b10  [5] profile_low  [4] profile_high
 *   profile≠3: [3] show_existing  [2] frame_type  [1] show_frame
 *   profile=3:  [3] reserved       [2] show_existing [1] frame_type [0] show_frame */
static bool vp9_show_frame(const uint8_t *data, size_t len)
{
    if (!data || len < 1) return true;
    uint8_t b = data[0];
    if ((b >> 6) != 2) return true;                   /* bad frame_marker, assume shown */
    int profile = ((b >> 5) & 1) | (((b >> 4) & 1) << 1);
    if (profile != 3) {
        if ((b >> 3) & 1) return true;                /* show_existing_frame */
        return (b >> 1) & 1;                          /* show_frame */
    } else {
        if ((b >> 2) & 1) return true;                /* show_existing_frame (profile 3) */
        return b & 1;                                 /* show_frame (profile 3) */
    }
}

/* For VP9 / HEVC / AV1: slice data is already a complete coded picture */
static VAStatus do_generic_decode(RKContext *c, RKDriver *d)
{
    uint8_t *pkt_data = NULL;
    size_t   pkt_sz   = 0;
    size_t   pkt_cap  = 0;

#define PKT_APPEND(ptr, len) do {                           \
    size_t _l = (len);                                      \
    if (pkt_sz + _l > pkt_cap) {                           \
        pkt_cap = (pkt_sz + _l) * 2 + 4096;               \
        pkt_data = realloc(pkt_data, pkt_cap);             \
        if (!pkt_data) return VA_STATUS_ERROR_ALLOCATION_FAILED; \
    }                                                       \
    memcpy(pkt_data + pkt_sz, (ptr), _l);                 \
    pkt_sz += _l;                                          \
} while (0)

    for (int i = 0; i < c->n_pending; i++) {
        RKBuffer *b = buffer_by_id(d, c->pending[i]);
        if (!b || b->type != VASliceDataBufferType) continue;
        PKT_APPEND(b->data, (size_t)b->size * b->num_elements);
    }
#undef PKT_APPEND

    if (!pkt_sz) {
        LOG("do_generic_decode: no slice data, marking surface 0x%x decoded",
            (unsigned)c->render_target);
        free(pkt_data);
        RKSurface *tgt = surface_by_id(d, c->render_target);
        if (tgt) {
            pthread_mutex_lock(&tgt->lock);
            tgt->decoded = true;
            pthread_cond_signal(&tgt->cond);
            pthread_mutex_unlock(&tgt->lock);
        }
        return VA_STATUS_SUCCESS;
    }

    /* Detect VP9 altref (show_frame=0): MPP decodes them as references but does
     * NOT output them via decode_get_frame in this environment.  The surface is
     * marked decoded immediately (placeholder prime_fd stays valid) and the job
     * is submitted with nosync=1 so nothing ever blocks for its output. */
    bool is_hidden = (c->coding == MPP_VIDEO_CodingVP9) &&
                     !vp9_show_frame(pkt_data, pkt_sz);

    if (is_hidden) {
        RKSurface *tgt = surface_by_id(d, c->render_target);
        if (tgt) {
            pthread_mutex_lock(&tgt->lock);
            tgt->decoded = true;
            pthread_cond_signal(&tgt->cond);
            pthread_mutex_unlock(&tgt->lock);
        }
    }

    LOG("do_generic_decode: submit %zu bytes to MPP (coding=%d) target=0x%x%s",
        pkt_sz, (int)c->coding, (unsigned)c->render_target,
        is_hidden ? " [altref]" : "");
    VAStatus st = rk_enqueue_job(c, pkt_data, pkt_sz, c->render_target, is_hidden);
    free(pkt_data);
    return st;
}

static VAStatus rk_EndPicture(VADriverContextP ctx, VAContextID ctx_id) {
    LOG("EndPicture: ctx=0x%x", ctx_id);
    RKDriver  *d = drv_from_ctx(ctx);
    RKContext *c = context_by_id(d, ctx_id);
    if (!c) return VA_STATUS_ERROR_INVALID_CONTEXT;

    VAStatus st;
    if (c->coding == MPP_VIDEO_CodingAVC)
        st = do_h264_decode(c, d);
    else
        st = do_generic_decode(c, d);

    return st;
}

static VAStatus rk_SyncSurface(VADriverContextP ctx, VASurfaceID id) {
    RKDriver  *d = drv_from_ctx(ctx);
    RKSurface *s = surface_by_id(d, id);
    if (!s) return VA_STATUS_ERROR_INVALID_SURFACE;

    pthread_mutex_lock(&s->lock);
    bool ready  = s->decoded || (s->ctx_id == 0);  /* not started = placeholder valid */
    pthread_mutex_unlock(&s->lock);

    if (ready) {
        LOG("SyncSurface: surface=0x%x prime_fd=%d ready", id, s->prime_fd);
        return VA_STATUS_SUCCESS;
    }

    /* Wait for the decode worker to deliver this surface's frame.  The worker
     * is the only thread that calls decode_get_frame, so no active draining
     * here.  Timeout is generous (1s); a genuine MPP ocasional drop at 4K is
     * resolved by the worker instantly (awaited ring), so a long wait here
     * means either MPP is briefly slow or this surface's AU was dropped. */
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += 1 * 1000 * 1000 * 1000;
    if (deadline.tv_nsec >= 1000000000) { deadline.tv_sec += 1; deadline.tv_nsec -= 1000000000; }

    pthread_mutex_lock(&s->lock);
    int done = s->decoded;
    while (!done) {
        int rc = pthread_cond_timedwait(&s->cond, &s->lock, &deadline);
        if (rc == ETIMEDOUT) {
            /* A frame that never arrived while the decoder is otherwise alive
             * is a hardware drop — but ONLY treat it as a dropped (stale)
             * frame if this surface already holds real pixels.  A surface
             * that was never filled must NOT be handed out (its placeholder
             * is neutral gray, but a "success" would still be a wrong frame);
             * report a real error there. */
            RKContext *cc = s->ctx_id ? context_by_id(d, s->ctx_id) : NULL;
            if (cc && cc->frames_out > 0 && s->priv_filled) {
                LOG("SyncSurface: DROPPED surface=0x%x (stale, frames=%lld)",
                    id, (long long)cc->frames_out);
                done = 1;
            } else {
                LOG("SyncSurface: TIMEOUT surface=0x%x prime_fd=%d filled=%d",
                    id, s->prime_fd, s->priv_filled ? 1 : 0);
            }
            break;
        }
        done = s->decoded;
    }
    pthread_mutex_unlock(&s->lock);

    LOG("SyncSurface: surface=0x%x %s prime_fd=%d",
        id, done ? "OK" : "TIMEOUT", s->prime_fd);
    return done ? VA_STATUS_SUCCESS : VA_STATUS_ERROR_DECODING_ERROR;
}

static VAStatus rk_SyncSurface2(VADriverContextP ctx,
                                  VASurfaceID id, uint64_t timeout_ns) {
    (void)timeout_ns;
    return rk_SyncSurface(ctx, id);
}

static VAStatus rk_QuerySurfaceStatus(VADriverContextP ctx,
                                       VASurfaceID id,
                                       VASurfaceStatus *status) {
    RKDriver  *d = drv_from_ctx(ctx);
    RKSurface *s = surface_by_id(d, id);
    if (!s) return VA_STATUS_ERROR_INVALID_SURFACE;
    pthread_mutex_lock(&s->lock);
    *status = s->decoded ? VASurfaceReady : VASurfaceRendering;
    pthread_mutex_unlock(&s->lock);
    LOG("QuerySurfaceStatus: surface=0x%x status=%s", id,
        (*status == VASurfaceReady) ? "Ready" : "Rendering");
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_ExportSurfaceHandle(VADriverContextP ctx,
                                        VASurfaceID id,
                                        uint32_t mem_type,
                                        uint32_t flags,
                                        void *descriptor) {
    RKDriver  *d = drv_from_ctx(ctx);
    RKSurface *s = surface_by_id(d, id);
    LOG("ExportSurfaceHandle: surface=0x%x mem_type=0x%x flags=0x%x",
        id, mem_type, flags);
    if (!s) return VA_STATUS_ERROR_INVALID_SURFACE;
    (void)flags;

    if (mem_type != VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2) {
        LOG("ExportSurfaceHandle: unsupported mem_type 0x%x", mem_type);
        return VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE;
    }

    /* If decode is in progress, sync now so the exported DMA-BUF contains the
     * correct frame. Firefox calls ExportSurfaceHandle before SyncSurface when
     * EndPicture is async; without this the EGLImage gets stale data. */
    pthread_mutex_lock(&s->lock);
    bool needs_sync = !s->decoded && (s->ctx_id != 0);
    pthread_mutex_unlock(&s->lock);
    if (needs_sync) {
        VAStatus st = rk_SyncSurface(ctx, id);
        if (st != VA_STATUS_SUCCESS) {
            LOG("ExportSurfaceHandle: front-sync failed gr=%d, abort export", (int)st);
            return st;
        }
    }

    pthread_mutex_lock(&s->lock);
    int fd       = (s->export_fd > 0) ? s->export_fd : s->prime_fd;
    int hs       = s->hstride ? s->hstride : s->width;
    int vs       = s->vstride ? s->vstride : s->height;
    bool decoded = s->decoded;
    bool is_placeholder = (s->priv_buf != NULL);
    bool is_10bit = MPP_FRAME_FMT_IS_YUV_10BIT(s->fmt);
    pthread_mutex_unlock(&s->lock);

    if (fd <= 0) {
        LOG("ExportSurfaceHandle: no fd (fd=%d decoded=%d), ERROR_INVALID_SURFACE", fd, decoded);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }

    int export_fd = dup(fd);
    if (export_fd < 0) {
        LOG("ExportSurfaceHandle: dup(%d) failed errno=%d, ERROR_ALLOCATION_FAILED", fd, errno);
        return VA_STATUS_ERROR_ALLOCATION_FAILED;
    }

    LOG("ExportSurfaceHandle: surface=0x%x %dx%d stride=%dx%d export_fd=%d decoded=%d placeholder=%d 10bit=%d",
        id, s->width, s->height, hs, vs, export_fd, decoded, is_placeholder, is_10bit);

    VADRMPRIMESurfaceDescriptor *desc = descriptor;
    memset(desc, 0, sizeof(*desc));
    desc->width       = (uint32_t)s->width;
    desc->height      = (uint32_t)s->height;
    desc->num_objects = 1;
    desc->objects[0].fd                  = export_fd;
    desc->objects[0].drm_format_modifier = 0; /* DRM_FORMAT_MOD_LINEAR */

    bool composed = (flags & VA_EXPORT_SURFACE_COMPOSED_LAYERS) != 0;

    /* COMPOSED_LAYERS: single NV12/P010 layer with 2 planes (mpv, GStreamer).
     * SEPARATE_LAYERS (default): R8/GR88 split planes (Firefox DMABufSurfaceYUV). */
    if (!is_10bit && composed) {
        desc->fourcc     = VA_FOURCC_NV12;
        desc->num_layers = 1;
        desc->objects[0].size           = (uint32_t)(hs * vs * 3 / 2);
        desc->layers[0].drm_format      = 0x3231564e; /* DRM_FORMAT_NV12 */
        desc->layers[0].num_planes      = 2;
        desc->layers[0].object_index[0] = 0;
        desc->layers[0].offset[0]       = 0;
        desc->layers[0].pitch[0]        = (uint32_t)hs;
        desc->layers[0].object_index[1] = 0;
        desc->layers[0].offset[1]       = (uint32_t)(hs * vs);
        desc->layers[0].pitch[1]        = (uint32_t)hs;
        return VA_STATUS_SUCCESS;
    }
    if (is_10bit && composed) {
        desc->fourcc     = VA_FOURCC_P010;
        desc->num_layers = 1;
        desc->objects[0].size           = (uint32_t)(hs * vs * 3);
        desc->layers[0].drm_format      = 0x30313050; /* DRM_FORMAT_P010 */
        desc->layers[0].num_planes      = 2;
        desc->layers[0].object_index[0] = 0;
        desc->layers[0].offset[0]       = 0;
        desc->layers[0].pitch[0]        = (uint32_t)(hs * 2);
        desc->layers[0].object_index[1] = 0;
        desc->layers[0].offset[1]       = (uint32_t)(hs * vs * 2);
        desc->layers[0].pitch[1]        = (uint32_t)(hs * 2);
        return VA_STATUS_SUCCESS;
    }

    desc->num_layers = 2;

    if (is_10bit) {
        /*
         * P010: 10-bit NV12 semi-planar.  Each luma/chroma sample is uint16.
         * Two-layer layout matching Firefox DMABufSurfaceYUV P010 import:
         *   layer 0 → DRM_FORMAT_R16    (Y,  16-bit luma, 10 significant bits)
         *   layer 1 → DRM_FORMAT_GR1616 (UV, 2×16-bit interleaved chroma)
         * Both layers share pitch = hs*2 (bytes per luma row).
         * UV offset = hs*vs*2 (Y plane size in bytes).
         */
        desc->fourcc                         = VA_FOURCC_P010;
        desc->objects[0].size                = (uint32_t)(hs * vs * 3);
        /* Y plane */
        desc->layers[0].drm_format           = 0x20363152; /* DRM_FORMAT_R16    */
        desc->layers[0].num_planes           = 1;
        desc->layers[0].object_index[0]      = 0;
        desc->layers[0].offset[0]            = 0;
        desc->layers[0].pitch[0]             = (uint32_t)(hs * 2);
        /* UV plane */
        desc->layers[1].drm_format           = 0x36315247; /* DRM_FORMAT_GR1616 */
        desc->layers[1].num_planes           = 1;
        desc->layers[1].object_index[0]      = 0;
        desc->layers[1].offset[0]            = (uint32_t)(hs * vs * 2);
        desc->layers[1].pitch[0]             = (uint32_t)(hs * 2);
    } else {
        /*
         * NV12: 8-bit YUV 4:2:0 semi-planar.
         * Firefox (DMABufSurfaceYUV) expects num_layers == plane count, each
         * layer being a single-plane view of the buffer:
         *   layer 0 → DRM_FORMAT_R8   (Y,  8-bit luma)
         *   layer 1 → DRM_FORMAT_GR88 (UV, interleaved chroma, 2 bytes/px)
         */
        desc->fourcc                         = VA_FOURCC_NV12;
        desc->objects[0].size                = (uint32_t)(hs * vs * 3 / 2);
        /* Y plane */
        desc->layers[0].drm_format           = 0x20203852; /* DRM_FORMAT_R8   */
        desc->layers[0].num_planes           = 1;
        desc->layers[0].object_index[0]      = 0;
        desc->layers[0].offset[0]            = 0;
        desc->layers[0].pitch[0]             = (uint32_t)hs;
        /* UV plane */
        desc->layers[1].drm_format           = 0x38385247; /* DRM_FORMAT_GR88 */
        desc->layers[1].num_planes           = 1;
        desc->layers[1].object_index[0]      = 0;
        desc->layers[1].offset[0]            = (uint32_t)(hs * vs);
        desc->layers[1].pitch[0]             = (uint32_t)hs;
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_QueryImageFormats(VADriverContextP ctx,
                                      VAImageFormat *list, int *n) {
    (void)ctx;
    list[0].fourcc         = VA_FOURCC_NV12;
    list[0].byte_order     = VA_LSB_FIRST;
    list[0].bits_per_pixel = 12;
    list[1].fourcc         = VA_FOURCC_P010;
    list[1].byte_order     = VA_LSB_FIRST;
    list[1].bits_per_pixel = 24;
    *n = 2;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_CreateImage(VADriverContextP ctx,
                                VAImageFormat *format,
                                int width, int height,
                                VAImage *image) {
    RKDriver *d = drv_from_ctx(ctx);
    (void)format;

    VABufferID buf_id;
    unsigned int stride = (unsigned int)((width + 15) & ~15);
    unsigned int size   = stride * (unsigned int)height * 3 / 2;
    VAStatus st = rk_CreateBuffer(ctx, 0, VAImageBufferType, size, 1,
                                  NULL, &buf_id);
    if (st != VA_STATUS_SUCCESS) {
        LOG("CreateImage: buffer alloc FAILED size=%u st=%d", size, (int)st);
        return st;
    }

    memset(image, 0, sizeof(*image));
    image->image_id    = buf_id; /* reuse buf_id as image_id for simplicity */
    image->buf         = buf_id;
    image->format      = *format;
    image->width       = (unsigned short)width;
    image->height      = (unsigned short)height;
    image->num_planes  = 2;
    image->pitches[0]  = stride;
    image->pitches[1]  = stride;
    image->offsets[0]  = 0;
    image->offsets[1]  = stride * (unsigned int)height;
    image->data_size   = size;
    (void)d;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_DeriveImage(VADriverContextP ctx,
                                VASurfaceID sid, VAImage *image) {
    RKDriver  *d = drv_from_ctx(ctx);
    RKSurface *s = surface_by_id(d, sid);
    if (!s) return VA_STATUS_ERROR_INVALID_SURFACE;

    /* priv_buf is not CPU-mappable as a shared VAImage; force callers to use
     * vaGetImage instead (copies pixels, but always works). */
    (void)image;
    return VA_STATUS_ERROR_OPERATION_FAILED;

    VAImageFormat fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.fourcc = VA_FOURCC_NV12;
    return rk_CreateImage(ctx, &fmt, s->width, s->height, image);
}

static VAStatus rk_DestroyImage(VADriverContextP ctx, VAImageID id) {
    return rk_DestroyBuffer(ctx, (VABufferID)id);
}

/* ── stub implementations ────────────────────────────────────── */

/* Suppress -Wunused-parameter for pure stub functions */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

static VAStatus rk_SetImagePalette(VADriverContextP ctx, VAImageID image,
                                    unsigned char *palette)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_GetImage(VADriverContextP ctx, VASurfaceID surface_id,
                             int x, int y, unsigned int w,
                             unsigned int h, VAImageID image_id)
{
    (void)x; (void)y; (void)w; (void)h;
    RKDriver  *d  = drv_from_ctx(ctx);
    RKSurface *s  = surface_by_id(d, surface_id);
    RKBuffer  *ib = buffer_by_id(d, (VABufferID)image_id);
    if (!s || !ib || !ib->data) {
        LOG("GetImage: surface=0x%x img=0x%x FAIL s=%d ib=%d data=%d",
            (unsigned)surface_id, (unsigned)image_id,
            s ? 1 : 0, ib ? 1 : 0, (ib && ib->data) ? 1 : 0);
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }

    int hs  = s->hstride ? s->hstride : s->width;
    int vs  = s->vstride ? s->vstride : s->height;
    bool i10 = MPP_FRAME_FMT_IS_YUV_10BIT(s->fmt);
    int bpp  = i10 ? 2 : 1;
    int img_hs = (int)(((unsigned int)s->width + 15u) & ~15u) * bpp;

    /* Under s->lock the pinned last_frame can't be deinit'd mid-read. */
    pthread_mutex_lock(&s->lock);
    const uint8_t *sp = NULL;
    if (s->last_frame) {
        MppBuffer lb = mpp_frame_get_buffer(s->last_frame);
        if (lb) sp = (const uint8_t *)mpp_buffer_get_ptr(lb);
    }
    if (!sp && s->priv_buf)
        sp = (const uint8_t *)mpp_buffer_get_ptr(s->priv_buf);
    if (!sp) {
        pthread_mutex_unlock(&s->lock);
        return VA_STATUS_ERROR_DECODING_ERROR;
    }
    uint8_t *dp = (uint8_t *)ib->data;
    for (int r = 0; r < s->height; r++)
        memcpy(dp + r * img_hs, sp + r * hs * bpp, (size_t)img_hs);
    const uint8_t *su = sp + (size_t)hs * vs * bpp;
    uint8_t       *du = dp + (size_t)img_hs * s->height;
    for (int r = 0; r < s->height / 2; r++)
        memcpy(du + r * img_hs, su + r * hs * bpp, (size_t)img_hs);
    pthread_mutex_unlock(&s->lock);
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_PutImage(VADriverContextP ctx, VASurfaceID surface,
                             VAImageID image, int src_x, int src_y,
                             unsigned int src_width, unsigned int src_height,
                             int dest_x, int dest_y,
                             unsigned int dest_width, unsigned int dest_height)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_QuerySubpicFmts(VADriverContextP ctx,
                                    VAImageFormat *format_list,
                                    unsigned int *flags,
                                    unsigned int *num_formats)
{ *num_formats = 0; return VA_STATUS_SUCCESS; }

static VAStatus rk_CreateSubpicture(VADriverContextP ctx,
                                     VAImageID image,
                                     VASubpictureID *subpicture)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_DestroySubpicture(VADriverContextP ctx,
                                      VASubpictureID subpicture)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_SetSubpicImage(VADriverContextP ctx,
                                   VASubpictureID subpicture, VAImageID image)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_SetSubpicChromakey(VADriverContextP ctx,
                                       VASubpictureID subpicture,
                                       unsigned int chromakey_min,
                                       unsigned int chromakey_max,
                                       unsigned int chromakey_mask)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_SetSubpicAlpha(VADriverContextP ctx,
                                   VASubpictureID subpicture, float global_alpha)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_AssociateSubpic(VADriverContextP ctx,
                                    VASubpictureID subpicture,
                                    VASurfaceID *target_surfaces, int num_surfaces,
                                    short src_x, short src_y,
                                    unsigned short src_width, unsigned short src_height,
                                    short dest_x, short dest_y,
                                    unsigned short dest_width, unsigned short dest_height,
                                    unsigned int flags)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_DeassociateSubpic(VADriverContextP ctx,
                                      VASubpictureID subpicture,
                                      VASurfaceID *target_surfaces, int num_surfaces)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_QueryDisplayAttrs(VADriverContextP ctx,
                                      VADisplayAttribute *attr_list,
                                      int *num_attributes)
{ *num_attributes = 0; return VA_STATUS_SUCCESS; }

static VAStatus rk_GetDisplayAttrs(VADriverContextP ctx,
                                    VADisplayAttribute *attr_list, int num_attributes)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_SetDisplayAttrs(VADriverContextP ctx,
                                    VADisplayAttribute *attr_list, int num_attributes)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_BufferInfo(VADriverContextP ctx, VABufferID buf_id,
                               VABufferType *type, unsigned int *size,
                               unsigned int *num_elements)
{
    RKDriver *d = drv_from_ctx(ctx);
    RKBuffer *b = buffer_by_id(d, buf_id);
    if (!b) return VA_STATUS_ERROR_INVALID_BUFFER;
    if (type)         *type         = b->type;
    if (size)         *size         = b->size;
    if (num_elements) *num_elements = b->num_elements;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_LockSurface(VADriverContextP ctx, VASurfaceID surface,
                                unsigned int *fourcc,
                                unsigned int *luma_stride,
                                unsigned int *chroma_u_stride,
                                unsigned int *chroma_v_stride,
                                unsigned int *luma_offset,
                                unsigned int *chroma_u_offset,
                                unsigned int *chroma_v_offset,
                                unsigned int *buffer_name, void **buffer)
{ return VA_STATUS_ERROR_UNIMPLEMENTED; }

static VAStatus rk_UnlockSurface(VADriverContextP ctx, VASurfaceID surface)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_QuerySurfaceAttrs(VADriverContextP ctx, VAConfigID config,
                                      VASurfaceAttrib *attrib_list,
                                      unsigned int *num_attribs)
{
    LOG("QuerySurfaceAttributes: config=0x%x list=%s",
        config, attrib_list ? "provided" : "NULL (query count)");

    /* Firefox calls this twice: first with NULL to get count, then with buffer */
    const unsigned int n = 7;
    if (!attrib_list) {
        *num_attribs = n;
        return VA_STATUS_SUCCESS;
    }
    if (*num_attribs < n) {
        *num_attribs = n;
        return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
    }

    /* Pixel format: NV12 (8-bit) */
    attrib_list[0].type              = VASurfaceAttribPixelFormat;
    attrib_list[0].flags             = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[0].value.type        = VAGenericValueTypeInteger;
    attrib_list[0].value.value.i     = VA_FOURCC_NV12;

    /* Pixel format: P010 (10-bit) */
    attrib_list[1].type              = VASurfaceAttribPixelFormat;
    attrib_list[1].flags             = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[1].value.type        = VAGenericValueTypeInteger;
    attrib_list[1].value.value.i     = VA_FOURCC_P010;

    /* Memory type: VA-managed + DRM PRIME 2 */
    attrib_list[2].type              = VASurfaceAttribMemoryType;
    attrib_list[2].flags             = VA_SURFACE_ATTRIB_GETTABLE |
                                       VA_SURFACE_ATTRIB_SETTABLE;
    attrib_list[2].value.type        = VAGenericValueTypeInteger;
    attrib_list[2].value.value.i     = (int)(VA_SURFACE_ATTRIB_MEM_TYPE_VA |
                                       VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2);

    /* Max/min resolution */
    attrib_list[3].type              = VASurfaceAttribMaxWidth;
    attrib_list[3].flags             = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[3].value.type        = VAGenericValueTypeInteger;
    attrib_list[3].value.value.i     = 7680;

    attrib_list[4].type              = VASurfaceAttribMaxHeight;
    attrib_list[4].flags             = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[4].value.type        = VAGenericValueTypeInteger;
    attrib_list[4].value.value.i     = 4320;

    attrib_list[5].type              = VASurfaceAttribMinWidth;
    attrib_list[5].flags             = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[5].value.type        = VAGenericValueTypeInteger;
    attrib_list[5].value.value.i     = 16;

    attrib_list[6].type              = VASurfaceAttribMinHeight;
    attrib_list[6].flags             = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[6].value.type        = VAGenericValueTypeInteger;
    attrib_list[6].value.value.i     = 16;

    *num_attribs = n;
    LOG("QuerySurfaceAttributes: returned %u attribs (NV12, P010, PRIME2, WxH)", n);
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_AcquireBufferHandle(VADriverContextP ctx,
                                        VABufferID buf_id, VABufferInfo *buf_info)
{ return VA_STATUS_ERROR_UNIMPLEMENTED; }

static VAStatus rk_ReleaseBufferHandle(VADriverContextP ctx, VABufferID buf_id)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_CreateMFContext(VADriverContextP ctx,
                                    VAMFContextID *mfe_context)
{ return VA_STATUS_ERROR_UNIMPLEMENTED; }

static VAStatus rk_MFAddContext(VADriverContextP ctx,
                                 VAMFContextID mf_context, VAContextID context)
{ return VA_STATUS_ERROR_UNIMPLEMENTED; }

static VAStatus rk_MFReleaseContext(VADriverContextP ctx,
                                     VAMFContextID mf_context, VAContextID context)
{ return VA_STATUS_ERROR_UNIMPLEMENTED; }

static VAStatus rk_MFSubmit(VADriverContextP ctx, VAMFContextID mf_context,
                             VAContextID *contexts, int num_contexts)
{ return VA_STATUS_ERROR_UNIMPLEMENTED; }

static VAStatus rk_QueryProcessingRate(VADriverContextP ctx,
                                        VAConfigID config_id,
                                        VAProcessingRateParameter *proc_buf,
                                        unsigned int *processing_rate)
{ return VA_STATUS_ERROR_UNIMPLEMENTED; }

static VAStatus rk_SyncBuffer(VADriverContextP ctx, VABufferID buf_id,
                               uint64_t timeout_ns)
{ return VA_STATUS_SUCCESS; }

static VAStatus rk_Copy(VADriverContextP ctx, VACopyObject *dst,
                         VACopyObject *src, VACopyOption option)
{ return VA_STATUS_ERROR_UNIMPLEMENTED; }

#pragma GCC diagnostic pop

static VAStatus rk_PutSurface(VADriverContextP ctx, VASurfaceID s,
                               void *draw, short sx, short sy,
                               unsigned short sw, unsigned short sh,
                               short dx, short dy,
                               unsigned short dw, unsigned short dh,
                               VARectangle *clips, unsigned int nc,
                               unsigned int flags) {
    (void)ctx;(void)s;(void)draw;(void)sx;(void)sy;(void)sw;(void)sh;
    (void)dx;(void)dy;(void)dw;(void)dh;(void)clips;(void)nc;(void)flags;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_QuerySurfaceError(VADriverContextP ctx, VASurfaceID s,
                                      VAStatus err, void **info) {
    (void)ctx;(void)s;(void)err; *info = NULL;
    return VA_STATUS_SUCCESS;
}

static VAStatus rk_CreateBuffer2(VADriverContextP ctx, VAContextID context,
                                  VABufferType type,
                                  unsigned int w, unsigned int h,
                                  unsigned int *unit_size,
                                  unsigned int *pitch,
                                  VABufferID *id) {
    unsigned int stride = (w + 15) & ~15;
    unsigned int size   = stride * h;
    if (unit_size) *unit_size = size;
    if (pitch)     *pitch     = stride;
    return rk_CreateBuffer(ctx, context, type, size, 1, NULL, id);
}

static VAStatus rk_GetSurfaceAttributes(VADriverContextP ctx,
                                         VAConfigID config,
                                         VASurfaceAttrib *list,
                                         unsigned int n) {
    (void)ctx;(void)config;(void)list;(void)n;
    return VA_STATUS_SUCCESS;
}

/* ── driver init ─────────────────────────────────────────────── */

VAStatus __vaDriverInit_1_20(VADriverContextP ctx)  /* NOLINT */
{
    log_init();
    LOG("__vaDriverInit_1_20: entry");
    RKDriver *d = calloc(1, sizeof(*d));
    if (!d) return VA_STATUS_ERROR_ALLOCATION_FAILED;
    ctx->pDriverData = d;
    pthread_mutex_init(&d->obj_mtx, NULL);

    ctx->version_major        = VA_MAJOR_VERSION;
    ctx->version_minor        = VA_MINOR_VERSION;
    ctx->max_profiles         = 16;
    ctx->max_entrypoints      = 4;
    ctx->max_attributes       = 8;
    ctx->max_image_formats    = 4;
    ctx->max_subpic_formats   = 4;
    ctx->max_display_attributes = 4;
    ctx->str_vendor           = "Rockchip MPP VA-API Driver 0.1";

    struct VADriverVTable *v = ctx->vtable;
    v->vaTerminate                = rk_Terminate;
    v->vaQueryConfigProfiles      = rk_QueryConfigProfiles;
    v->vaQueryConfigEntrypoints   = rk_QueryConfigEntrypoints;
    v->vaGetConfigAttributes      = rk_GetConfigAttributes;
    v->vaCreateConfig             = rk_CreateConfig;
    v->vaDestroyConfig            = rk_DestroyConfig;
    v->vaQueryConfigAttributes    = rk_QueryConfigAttributes;
    v->vaCreateSurfaces           = rk_CreateSurfaces;
    v->vaDestroySurfaces          = rk_DestroySurfaces;
    v->vaCreateContext            = rk_CreateContext;
    v->vaDestroyContext           = rk_DestroyContext;
    v->vaCreateBuffer             = rk_CreateBuffer;
    v->vaBufferSetNumElements     = rk_BufferSetNumElements;
    v->vaMapBuffer                = rk_MapBuffer;
    v->vaUnmapBuffer              = rk_UnmapBuffer;
    v->vaDestroyBuffer            = rk_DestroyBuffer;
    v->vaBeginPicture             = rk_BeginPicture;
    v->vaRenderPicture            = rk_RenderPicture;
    v->vaEndPicture               = rk_EndPicture;
    v->vaSyncSurface              = rk_SyncSurface;
    v->vaQuerySurfaceStatus       = rk_QuerySurfaceStatus;
    v->vaQuerySurfaceError        = rk_QuerySurfaceError;
    v->vaPutSurface               = rk_PutSurface;
    v->vaQueryImageFormats        = rk_QueryImageFormats;
    v->vaCreateImage              = rk_CreateImage;
    v->vaDeriveImage              = rk_DeriveImage;
    v->vaDestroyImage             = rk_DestroyImage;
    v->vaSetImagePalette          = rk_SetImagePalette;
    v->vaGetImage                 = rk_GetImage;
    v->vaPutImage                 = rk_PutImage;
    v->vaQuerySubpictureFormats   = rk_QuerySubpicFmts;
    v->vaCreateSubpicture         = rk_CreateSubpicture;
    v->vaDestroySubpicture        = rk_DestroySubpicture;
    v->vaSetSubpictureImage       = rk_SetSubpicImage;
    v->vaSetSubpictureChromakey   = rk_SetSubpicChromakey;
    v->vaSetSubpictureGlobalAlpha = rk_SetSubpicAlpha;
    v->vaAssociateSubpicture      = rk_AssociateSubpic;
    v->vaDeassociateSubpicture    = rk_DeassociateSubpic;
    v->vaQueryDisplayAttributes   = rk_QueryDisplayAttrs;
    v->vaGetDisplayAttributes     = rk_GetDisplayAttrs;
    v->vaSetDisplayAttributes     = rk_SetDisplayAttrs;
    v->vaBufferInfo               = rk_BufferInfo;
    v->vaLockSurface              = rk_LockSurface;
    v->vaUnlockSurface            = rk_UnlockSurface;
    v->vaGetSurfaceAttributes     = rk_GetSurfaceAttributes;
    v->vaCreateSurfaces2          = rk_CreateSurfaces2;
    v->vaQuerySurfaceAttributes   = rk_QuerySurfaceAttrs;
    v->vaAcquireBufferHandle      = rk_AcquireBufferHandle;
    v->vaReleaseBufferHandle      = rk_ReleaseBufferHandle;
    v->vaCreateMFContext          = rk_CreateMFContext;
    v->vaMFAddContext             = rk_MFAddContext;
    v->vaMFReleaseContext         = rk_MFReleaseContext;
    v->vaMFSubmit                 = rk_MFSubmit;
    v->vaCreateBuffer2            = rk_CreateBuffer2;
    v->vaQueryProcessingRate      = rk_QueryProcessingRate;
    v->vaExportSurfaceHandle      = rk_ExportSurfaceHandle;
    v->vaSyncSurface2             = rk_SyncSurface2;
    v->vaSyncBuffer               = rk_SyncBuffer;
    v->vaCopy                     = rk_Copy;

    LOG("driver init OK — Rockchip RK3588 MPP");
    return VA_STATUS_SUCCESS;
}

/* Compatibility with older libva (e.g. libva 2.14 / VA-API 1.14 shipped in
 * Debian 10-based UOS 20): libva only probes __vaDriverInit_<major>_<minor>
 * down from its OWN minor version and never tries __vaDriverInit_1_20.
 * Export a 1_14 alias of the same entry point so the driver loads there too. */
VAStatus __vaDriverInit_1_14(VADriverContextP ctx) __attribute__((alias("__vaDriverInit_1_20")));
