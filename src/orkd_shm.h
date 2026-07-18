/* orkd_shm.h — Linux transport helpers for orkd's #2b-2 dma-buf sharing: SCM_RIGHTS fd passing over the
 * control socket + dma-heap allocation on the client side + a shared fnv. Separate from orkd_proto.h (the
 * wire structs) so the protocol header stays light. Included by src/orkd.c and src/orkd_client.c. */
#ifndef ORKD_SHM_H
#define ORKD_SHM_H

#include "orkd_proto.h"
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* FNV-1a over a byte buffer — cheap content hash both ends compute to confirm the shared buffer matches. */
static inline uint64_t orkd_fnv(const void *p, size_t n){
    const uint8_t *b = (const uint8_t *)p; uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++){ h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* Send one orkd_hdr (16B — atomic on a stream socket at this size) with an optional SCM_RIGHTS fd. 0/-1. */
static inline int orkd_send_hdr_fd(int sock, const struct orkd_hdr *h, int fd){
    struct iovec iov; iov.iov_base = (void *)h; iov.iov_len = sizeof *h;
    char cbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr msg; memset(&msg, 0, sizeof msg); msg.msg_iov = &iov; msg.msg_iovlen = 1;
    if (fd >= 0){
        msg.msg_control = cbuf; msg.msg_controllen = sizeof cbuf;
        struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
        c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(c), &fd, sizeof fd);
    }
    ssize_t r; while ((r = sendmsg(sock, &msg, 0)) < 0 && errno == EINTR){}
    return r == (ssize_t)sizeof *h ? 0 : -1;
}

/* recvmsg exactly one orkd_hdr, capturing an optional SCM_RIGHTS fd into *fdout (-1 if none). 1 ok / 0 EOF / -1. */
static inline int orkd_recv_hdr_fd(int sock, struct orkd_hdr *h, int *fdout){
    *fdout = -1;
    char *p = (char *)h; size_t need = sizeof *h, got = 0;
    while (got < need){
        char cbuf[CMSG_SPACE(sizeof(int))];
        struct iovec iov; iov.iov_base = p + got; iov.iov_len = need - got;
        struct msghdr msg; memset(&msg, 0, sizeof msg);
        msg.msg_iov = &iov; msg.msg_iovlen = 1; msg.msg_control = cbuf; msg.msg_controllen = sizeof cbuf;
        ssize_t r = recvmsg(sock, &msg, 0);
        if (r == 0) return 0;
        if (r < 0){ if (errno == EINTR) continue; return -1; }
        for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
            if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS){
                int f; memcpy(&f, CMSG_DATA(c), sizeof f); if (*fdout >= 0) close(*fdout); *fdout = f;
            }
        got += (size_t)r;
    }
    return 1;
}

#ifdef __linux__
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h>
/* Allocate a CPU-mappable dma-buf from /dev/dma_heap/system; returns the dma-buf fd or -1 (heap absent/full). */
static inline int orkd_dmaheap_alloc(size_t size){
    int hf = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (hf < 0) return -1;
    struct dma_heap_allocation_data d; memset(&d, 0, sizeof d);
    d.len = size; d.fd_flags = O_RDWR | O_CLOEXEC;
    int rc = ioctl(hf, DMA_HEAP_IOCTL_ALLOC, &d);
    close(hf);
    return rc == 0 ? (int)d.fd : -1;
}
/* dma-buf CPU cache sync (cross-process coherency for the cacheable system heap). Defs mirror the uABI. */
#ifndef DMA_BUF_IOCTL_SYNC
struct dma_buf_sync { uint64_t flags; };
#define DMA_BUF_SYNC_READ  (1 << 0)
#define DMA_BUF_SYNC_WRITE (2 << 0)
#define DMA_BUF_SYNC_START (0 << 2)
#define DMA_BUF_SYNC_END   (1 << 2)
#define DMA_BUF_IOCTL_SYNC _IOW('b', 0, struct dma_buf_sync)
#endif
/* Clean CPU writes -> device: call after the producer fills a buffer, before the NPU reads it (A/weights). */
static inline void orkd_dmabuf_clean(int fd){ struct dma_buf_sync s; s.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE; ioctl(fd, DMA_BUF_IOCTL_SYNC, &s); }
/* Invalidate CPU cache <- device: call before the consumer reads a buffer the NPU just wrote (C/output). */
static inline void orkd_dmabuf_invalidate(int fd){ struct dma_buf_sync s; s.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ; ioctl(fd, DMA_BUF_IOCTL_SYNC, &s); }
#endif /* __linux__ */

#endif /* ORKD_SHM_H */
