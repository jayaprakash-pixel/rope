/* Hardware probe (Linux). Same facts as probe.c, read from /proc, /sys and the NVIDIA driver library at run time.
 * No root, no CUDA toolkit. A missing piece is reported as absent, never as an error.
 * Nothing in the report names the user: of a path only "root filesystem" or "mounted drive" stays. */
#ifdef __linux__
/* not _GNU_SOURCE: it makes glibc 2.38+ link strtol/sscanf as __isoc23_*, and the program would then need glibc 2.38 */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "probe.h"
#include "kernels.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <dlfcn.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/sysmacros.h>
#include <sys/utsname.h>
#include <strings.h>
#ifndef O_DIRECT
#define O_DIRECT 040000   /* x86 and arm64 Linux value; the header hides it without _GNU_SOURCE */
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define ROPE_X86 1
#endif

/* ---------------- small helpers ---------------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }

/* Read a small text file; strip the trailing newline. Returns 0 when the file is missing or empty. */
static int rd(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "r");
    if (!f) { buf[0] = 0; return 0; }
    size_t n = fread(buf, 1, cap - 1, f); fclose(f);
    buf[n] = 0;
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) buf[--n] = 0;
    return n > 0;
}

static int placeholder(const char *s) {
    return !*s || !strcasecmp(s, "none") || !strcasecmp(s, "default string") || !strcasecmp(s, "to be filled by o.e.m.") ||
           !strcasecmp(s, "system product name") || !strcasecmp(s, "system version") || !strcasecmp(s, "not specified");
}

/* ---------------- CPU ---------------- */
static void probe_cpu(rope_probe *p) {
#ifdef ROPE_X86
    unsigned a, b, c, d;
    if (__get_cpuid_max(0x80000000, NULL) >= 0x80000004) {
        unsigned r[12];
        for (unsigned i = 0; i < 3; i++) __cpuid(0x80000002 + i, r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
        memcpy(p->cpu_name, r, 48); p->cpu_name[48] = 0;
        char *s = p->cpu_name; while (*s == ' ') s++;
        memmove(p->cpu_name, s, strlen(s) + 1);
        for (size_t n = strlen(p->cpu_name); n && p->cpu_name[n - 1] == ' '; n--) p->cpu_name[n - 1] = 0;
    }
    unsigned max_leaf = __get_cpuid_max(0, NULL);
    __cpuid(1, a, b, c, d);
    unsigned ecx1 = c;
    int osxsave = (ecx1 >> 27) & 1;
    unsigned long long xcr0 = 0;
    if (osxsave) { unsigned lo, hi; __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0)); xcr0 = ((unsigned long long)hi << 32) | lo; }
    int os_avx = (xcr0 & 6) == 6, os_avx512 = (xcr0 & 0xE6) == 0xE6;
    p->sse42 = (ecx1 >> 20) & 1;
    p->avx = ((ecx1 >> 28) & 1) && os_avx;
    p->fma = ((ecx1 >> 12) & 1) && os_avx;
    p->f16c = ((ecx1 >> 29) & 1) && os_avx;
    if (max_leaf >= 7) {
        __cpuid_count(7, 0, a, b, c, d);
        p->avx2 = ((b >> 5) & 1) && os_avx;
        p->avx512f = ((b >> 16) & 1) && os_avx512;
        p->avx512_vnni = ((c >> 11) & 1) && os_avx512;
        __cpuid_count(7, 1, a, b, c, d);
        p->avx_vnni = ((a >> 4) & 1) && os_avx;
    }
#endif
    /* threads, and cores = distinct (physical id, core id) pairs in /proc/cpuinfo */
    FILE *f = fopen("/proc/cpuinfo", "r");
    int pid_[512], cid_[512], npair = 0, cur_p = 0, cur_c = -1, procs = 0;
    char line[512];
    while (f && fgets(line, sizeof line, f)) {
        if (!strncmp(line, "processor", 9)) {
            if (procs && cur_c >= 0 && npair < 512) { pid_[npair] = cur_p; cid_[npair++] = cur_c; }
            procs++; cur_p = 0; cur_c = -1;
        } else if (!strncmp(line, "physical id", 11)) { const char *v = strchr(line, ':'); if (v) cur_p = atoi(v + 1); }
        else if (!strncmp(line, "core id", 7)) { const char *v = strchr(line, ':'); if (v) cur_c = atoi(v + 1); }
        else if (!p->cpu_name[0] && (!strncmp(line, "model name", 10) || !strncmp(line, "Model", 5))) {
            const char *v = strchr(line, ':');
            if (v) { v++; while (*v == ' ') v++; snprintf(p->cpu_name, sizeof p->cpu_name, "%s", v); p->cpu_name[strcspn(p->cpu_name, "\n")] = 0; }
        }
    }
    if (procs && cur_c >= 0 && npair < 512) { pid_[npair] = cur_p; cid_[npair++] = cur_c; }
    if (f) fclose(f);
    int cores = 0;
    for (int i = 0; i < npair; i++) { int dup = 0; for (int j = 0; j < i; j++) if (pid_[j] == pid_[i] && cid_[j] == cid_[i]) { dup = 1; break; } if (!dup) cores++; }
    long onl = sysconf(_SC_NPROCESSORS_ONLN);
    p->threads = onl > 0 ? (int)onl : procs;
    p->cores = cores > 0 ? cores : p->threads;
}

/* ---------------- NVIDIA via the driver library (libcuda.so.1), loaded at run time ---------------- */
typedef int (*cu_i)(unsigned);
typedef int (*cu_pi)(int *);
typedef int (*cu_get)(int *, int);
typedef int (*cu_name)(char *, int, int);
typedef int (*cu_mem)(size_t *, int);
typedef int (*cu_attr)(int *, int, int);
typedef int (*cu_ctx)(void **, unsigned, int);
typedef int (*cu_ctxd)(void *);
typedef int (*cu_info)(size_t *, size_t *);
typedef int (*cu_hostalloc)(void **, size_t, unsigned);
typedef int (*cu_hostfree)(void *);
typedef int (*cu_alloc)(unsigned long long *, size_t);
typedef int (*cu_free)(unsigned long long);
typedef int (*cu_h2d)(unsigned long long, const void *, size_t);

static double probe_h2d(void *h) {
    cu_hostalloc ha = (cu_hostalloc)dlsym(h, "cuMemHostAlloc");
    cu_hostfree hf = (cu_hostfree)dlsym(h, "cuMemFreeHost");
    cu_alloc da = (cu_alloc)dlsym(h, "cuMemAlloc_v2");
    cu_free df = (cu_free)dlsym(h, "cuMemFree_v2");
    cu_h2d cp = (cu_h2d)dlsym(h, "cuMemcpyHtoD_v2");
    const size_t n = 64u << 20;
    void *host = NULL; unsigned long long dev = 0; double best = 0;
    if (!ha || !hf || !da || !df || !cp || ha(&host, n, 0)) return 0;
    if (!da(&dev, n)) {
        memset(host, 1, n);
        for (int i = 0; i < 4; i++) {
            double a = now_s();
            if (cp(dev, host, n)) { best = 0; break; }
            double gbs = n / (now_s() - a) / 1e9;
            if (i > 0 && gbs > best) best = gbs;
        }
        df(dev);
    }
    hf(host);
    return best;
}

static void probe_cuda(rope_probe *p) {
    const char *lib = getenv("ROPE_NVCUDA_DLL");     /* diagnostics, and tests of a machine with no driver */
    void *h = dlopen(lib && *lib ? lib : "libcuda.so.1", RTLD_NOW);
    if (!h) return;
    cu_i init = (cu_i)dlsym(h, "cuInit");
    cu_pi count = (cu_pi)dlsym(h, "cuDeviceGetCount"), drv = (cu_pi)dlsym(h, "cuDriverGetVersion");
    cu_get get = (cu_get)dlsym(h, "cuDeviceGet");
    cu_name name = (cu_name)dlsym(h, "cuDeviceGetName");
    cu_mem mem = (cu_mem)dlsym(h, "cuDeviceTotalMem_v2");
    cu_attr attr = (cu_attr)dlsym(h, "cuDeviceGetAttribute");
    cu_ctx ctx = (cu_ctx)dlsym(h, "cuCtxCreate_v2");
    cu_ctxd ctxd = (cu_ctxd)dlsym(h, "cuCtxDestroy_v2");
    cu_info info = (cu_info)dlsym(h, "cuMemGetInfo_v2");
    int n = 0, dev = 0;
    if (!init || !count || !get || !name || !mem || !attr || init(0) || count(&n) || n < 1 || get(&dev, 0)) { dlclose(h); return; }
    size_t total = 0;
    name(p->cuda_name, sizeof p->cuda_name, dev);
    mem(&total, dev);
    attr(&p->cc_major, 75, dev);
    attr(&p->cc_minor, 76, dev);
    attr(&p->sm_count, 16, dev);
    attr(&p->clock_khz, 13, dev);
    if (drv) drv(&p->cuda_driver);
    p->vram_total = total;
    void *c = NULL;
    if (ctx && ctxd && info && !ctx(&c, 0, dev)) {
        size_t f = 0, t = 0;
        if (!info(&f, &t)) p->vram_free = f;
        p->h2d_gbs = probe_h2d(h);
        ctxd(c);
    }
    p->cuda_ok = 1;                                   /* the library stays loaded: unloading after cuInit is not supported */
}

/* ---------------- display adapters: /sys/class/drm ---------------- */
static void pci_lookup(int vendor, int device, char *out, size_t cap) {
    static const char *files[] = {"/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids", "/usr/share/pci.ids"};
    char vname[96] = "", dname[96] = "";
    for (int k = 0; k < 3 && !dname[0]; k++) {
        FILE *f = fopen(files[k], "r");
        if (!f) continue;
        char line[256]; int in_vendor = 0;
        while (fgets(line, sizeof line, f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            if (line[0] != '\t') {
                in_vendor = 0;
                unsigned v;
                if (sscanf(line, "%x", &v) == 1 && (int)v == vendor) {
                    in_vendor = 1; const char *s = line + 4; while (*s == ' ') s++;
                    snprintf(vname, sizeof vname, "%s", s); vname[strcspn(vname, "\n")] = 0;
                }
            } else if (in_vendor && line[1] != '\t') {
                unsigned d;
                if (sscanf(line + 1, "%x", &d) == 1 && (int)d == device) {
                    const char *s = line + 5; while (*s == ' ') s++;
                    snprintf(dname, sizeof dname, "%s", s); dname[strcspn(dname, "\n")] = 0; break;
                }
            }
        }
        fclose(f);
    }
    if (dname[0]) snprintf(out, cap, "%s %s", vname[0] ? vname : "", dname);
    else snprintf(out, cap, "%s GPU (device 0x%04X)", vendor == 0x10DE ? "NVIDIA" : vendor == 0x1002 ? "AMD" : vendor == 0x8086 ? "Intel" : "Unknown", device);
}

static void probe_adapters(rope_probe *p) {
    DIR *d = opendir("/sys/class/drm");
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) && p->n_adapters < ROPE_MAX_GPUS) {
        if (strncmp(e->d_name, "card", 4) || strchr(e->d_name, '-')) continue;
        char path[300], v[32], dv[32];
        snprintf(path, sizeof path, "/sys/class/drm/%s/device/vendor", e->d_name);
        if (!rd(path, v, sizeof v)) continue;
        snprintf(path, sizeof path, "/sys/class/drm/%s/device/device", e->d_name);
        rd(path, dv, sizeof dv);
        rope_adapter *r = &p->adapters[p->n_adapters++];
        memset(r, 0, sizeof *r);
        r->vendor = (int)strtol(v, NULL, 16);
        pci_lookup(r->vendor, (int)strtol(dv, NULL, 16), r->name, sizeof r->name);
        snprintf(path, sizeof path, "/sys/class/drm/%s/device/mem_info_vram_total", e->d_name);
        char m[40];
        if (rd(path, m, sizeof m)) r->dedicated_bytes = strtoull(m, NULL, 10);                 /* amdgpu */
        else if (r->vendor == 0x10DE && p->cuda_ok) r->dedicated_bytes = p->vram_total;
        r->shared_bytes = p->ram_total / 2;
        r->integrated = r->vendor == 0x8086 || (r->vendor != 0x10DE && r->dedicated_bytes < (512ull << 20));
    }
    closedir(d);
}

/* ---------------- storage ---------------- */
/* The mount point of the file system that holds dir: only "root filesystem" or "mounted drive" is reported. */
static void drive_only(const char *real, char *out, size_t cap) {
    FILE *f = fopen("/proc/self/mountinfo", "r");
    char line[1024], best[PATH_MAX] = "";
    while (f && fgets(line, sizeof line, f)) {
        char *s = line; int field = 0; char *mp = NULL;
        for (; *s && field < 5; s++) if (*s == ' ' && ++field == 4) mp = s + 1;
        if (!mp) continue;
        char *end = strchr(mp, ' '); if (!end) continue; *end = 0;
        size_t n = strlen(mp);
        if (!strncmp(real, mp, n) && (real[n] == '/' || real[n] == 0 || n == 1) && n > strlen(best)) snprintf(best, sizeof best, "%s", mp);
    }
    if (f) fclose(f);
    snprintf(out, cap, "%s", !strcmp(best, "/") || !best[0] ? "root filesystem" : "mounted drive");
}

static int has_file(const char *dir, const char *name) { char p[PATH_MAX + 64]; snprintf(p, sizeof p, "%s/%s", dir, name); return access(p, F_OK) == 0; }

static void probe_disk_type(rope_probe *p, const char *dir) {
    p->disk_ssd = -1; strcpy(p->disk_bus, "unknown");
    struct stat st;
    if (stat(dir, &st)) return;
    char link[PATH_MAX], real[PATH_MAX];
    snprintf(link, sizeof link, "/sys/dev/block/%u:%u", major(st.st_dev), minor(st.st_dev));
    if (!realpath(link, real)) return;
    const char *bus = "unknown";
    if (strstr(real, "/usb")) bus = "USB"; else if (strstr(real, "/nvme")) bus = "NVMe"; else if (strstr(real, "/ata")) bus = "SATA";
    else if (strstr(real, "/mmc")) bus = "MMC"; else if (strstr(real, "/virtio")) bus = "Virtual";
    /* partitions and LVM / encrypted volumes: climb to the physical disk that has queue/rotational */
    for (int depth = 0; depth < 4; depth++) {
        if (has_file(real, "partition")) { char *sl = strrchr(real, '/'); if (sl) *sl = 0; continue; }
        if (has_file(real, "slaves")) {
            char sp[PATH_MAX + 16]; snprintf(sp, sizeof sp, "%s/slaves", real);
            DIR *d = opendir(sp); struct dirent *e; int moved = 0;
            while (d && (e = readdir(d))) if (e->d_name[0] != '.') { char nx[PATH_MAX * 2]; snprintf(nx, sizeof nx, "%s/%s", sp, e->d_name); if (realpath(nx, real)) { moved = 1; break; } }
            if (d) closedir(d);
            if (moved) { if (!strcmp(bus, "unknown")) { if (strstr(real, "/usb")) bus = "USB"; else if (strstr(real, "/nvme")) bus = "NVMe"; else if (strstr(real, "/ata")) bus = "SATA"; } continue; }
        }
        break;
    }
    char rp[PATH_MAX + 32], v[16];
    snprintf(rp, sizeof rp, "%s/queue/rotational", real);
    if (rd(rp, v, sizeof v)) p->disk_ssd = v[0] == '0';
    snprintf(p->disk_bus, sizeof p->disk_bus, "%s", bus);
}

/* Write a temp file, read it back with the page cache bypassed (O_DIRECT; on a file system without it: fsync + drop pages). */
static void probe_disk_speed(rope_probe *p, const char *dir, int mb) {
    const size_t BLK = 4u << 20;
    char path[PATH_MAX + 64];
    snprintf(path, sizeof path, "%s/rope_probe_%d.tmp", dir, (int)getpid());
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, BLK)) return;
    unsigned char *b = buf;
    for (size_t i = 0; i < BLK; i++) b[i] = (unsigned char)(i * 2654435761u >> 24);
    int nblk = mb / 4, direct = 1, ok = 1;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_SYNC, 0600);
    if (fd < 0) { direct = 0; fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600); }
    if (fd < 0) { free(buf); return; }
    double t0 = now_s();
    for (int i = 0; i < nblk && ok; i++) ok = write(fd, buf, BLK) == (ssize_t)BLK;
    if (!direct) fsync(fd);
    double t1 = now_s();
    close(fd);
    if (ok) p->disk_write_gbs = (double)nblk * BLK / 1e9 / (t1 - t0);
    fd = ok ? open(path, O_RDONLY | (direct ? O_DIRECT : 0)) : -1;
    if (fd >= 0) {
        if (!direct) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        t0 = now_s();
        for (int i = 0; i < nblk && ok; i++) ok = read(fd, buf, BLK) == (ssize_t)BLK;
        t1 = now_s();
        close(fd);
        if (ok) p->disk_read_gbs = (double)nblk * BLK / 1e9 / (t1 - t0);
    }
    unlink(path);
    free(buf);
}

/* ---------------- system: OS, maker and model, CPU clock, power ---------------- */
static void probe_system(rope_probe *p) {
    char line[256] = "";
    FILE *f = fopen("/etc/os-release", "r");
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, "PRETTY_NAME=", 12)) {
            char *v = line + 12; if (*v == '"') v++;
            v[strcspn(v, "\"\n")] = 0;
            snprintf(p->os_name, sizeof p->os_name, "%s", v);
            break;
        }
    if (f) fclose(f);
    struct utsname u;
    if (!uname(&u)) snprintf(p->os, sizeof p->os, "%s %s", u.sysname, u.release);
    char maker[64], model[64], ver[64];
    rd("/sys/class/dmi/id/sys_vendor", maker, sizeof maker);
    rd("/sys/class/dmi/id/product_name", model, sizeof model);
    rd("/sys/class/dmi/id/product_version", ver, sizeof ver);
    if (placeholder(maker)) maker[0] = 0;
    if (placeholder(model)) model[0] = 0;
    if (!placeholder(ver) && model[0] && strstr(maker, "LENOVO")) snprintf(p->machine, sizeof p->machine, "%s %s (%s)", maker, model, ver);   /* Lenovo: product_name is a code */
    else snprintf(p->machine, sizeof p->machine, "%s%s%s", maker, maker[0] && model[0] ? " " : "", model);
    if (!p->machine[0]) snprintf(p->machine, sizeof p->machine, "unknown (a virtual machine or no firmware information)");
    char s[64];
    if (rd("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency", s, sizeof s) || rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", s, sizeof s)) p->cpu_mhz = atoi(s) / 1000;
    /* power */
    p->on_ac = -1; p->battery_pct = -1; p->battery_saver = -1;
    DIR *d = opendir("/sys/class/power_supply");
    struct dirent *e; int bat = 0, mains = 0, mains_on = 0, discharging = 0;
    while (d && (e = readdir(d))) {
        char pa[300], t[32], v[32];
        if (e->d_name[0] == '.') continue;
        snprintf(pa, sizeof pa, "/sys/class/power_supply/%s/type", e->d_name);
        if (!rd(pa, t, sizeof t)) continue;
        if (!strcmp(t, "Battery")) {
            snprintf(pa, sizeof pa, "/sys/class/power_supply/%s/scope", e->d_name);
            if (rd(pa, v, sizeof v) && !strcmp(v, "Device")) continue;        /* a mouse or phone battery, not the laptop's */
            bat = 1;
            snprintf(pa, sizeof pa, "/sys/class/power_supply/%s/capacity", e->d_name);
            if (rd(pa, v, sizeof v)) p->battery_pct = atoi(v);
            snprintf(pa, sizeof pa, "/sys/class/power_supply/%s/status", e->d_name);
            if (rd(pa, v, sizeof v) && !strcmp(v, "Discharging")) discharging = 1;
        } else if (!strcmp(t, "Mains")) {
            mains = 1;
            snprintf(pa, sizeof pa, "/sys/class/power_supply/%s/online", e->d_name);
            if (rd(pa, v, sizeof v) && v[0] == '1') mains_on = 1;
        }
    }
    if (d) closedir(d);
    if (bat) p->on_ac = mains ? mains_on : !discharging;
}

int rope_probe_run(rope_probe *p, const char *dir, int disk_test_mb) {
    memset(p, 0, sizeof *p);
    probe_system(p);
    char line[128];
    FILE *f = fopen("/proc/meminfo", "r");
    while (f && fgets(line, sizeof line, f)) {
        unsigned long long kb;
        if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) p->ram_total = kb * 1024;
        else if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) p->ram_free = kb * 1024;
    }
    if (f) fclose(f);
    /* installed RAM is not readable without root: MemTotal rounded up to the next 0.5 GiB (the kernel and firmware keep a small part) */
    { const uint64_t H = 512ull << 20; p->ram_installed = (p->ram_total + H - 1) / H * H; }
    probe_cpu(p);
    probe_cuda(p);
    probe_adapters(p);
    char real[PATH_MAX];
    const char *d = dir && *dir ? dir : ".";
    if (!realpath(d, real)) return -1;
    drive_only(real, p->disk_dir, sizeof p->disk_dir);
    struct statvfs sv;
    if (!statvfs(real, &sv)) { p->disk_free = (uint64_t)sv.f_bavail * sv.f_frsize; p->disk_total = (uint64_t)sv.f_blocks * sv.f_frsize; }
    probe_disk_type(p, real);
    if (disk_test_mb >= 64) probe_disk_speed(p, real, disk_test_mb);
    return 0;
}
#endif
