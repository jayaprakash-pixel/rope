/* Hardware probe (Windows). Uses only OS/driver interfaces that exist on any machine:
 * no CUDA toolkit, no admin rights. A missing piece is reported as absent, never as an error. */
#include "probe.h"
#include "kernels.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define COBJMACROS
#include <windows.h>
#include <winioctl.h>
#include <dxgi.h>
#include <intrin.h>
#endif

#ifdef _WIN32
/* ---------------- CPU (CPUID + OS support check through XGETBV) ---------------- */
static void probe_cpu(rope_probe *p) {
    int r[4];
    __cpuid(r, 0x80000000);
    if ((unsigned)r[0] >= 0x80000004) {
        int b[12];
        __cpuid(b, 0x80000002); __cpuid(b + 4, 0x80000003); __cpuid(b + 8, 0x80000004);
        memcpy(p->cpu_name, b, 48); p->cpu_name[48] = 0;
        char *s = p->cpu_name; while (*s == ' ') s++;
        memmove(p->cpu_name, s, strlen(s) + 1);
        for (size_t n = strlen(p->cpu_name); n && p->cpu_name[n - 1] == ' '; n--) p->cpu_name[n - 1] = 0;
    }
    __cpuid(r, 0);
    int max_leaf = r[0];
    __cpuid(r, 1);
    int ecx1 = r[2];
    int osxsave = (ecx1 >> 27) & 1;
    unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
    int os_avx = (xcr0 & 6) == 6, os_avx512 = (xcr0 & 0xE6) == 0xE6;
    p->sse42 = (ecx1 >> 20) & 1;
    p->avx = ((ecx1 >> 28) & 1) && os_avx;
    p->fma = ((ecx1 >> 12) & 1) && os_avx;
    p->f16c = ((ecx1 >> 29) & 1) && os_avx;
    if (max_leaf >= 7) {
        __cpuidex(r, 7, 0);
        p->avx2 = ((r[1] >> 5) & 1) && os_avx;
        p->avx512f = ((r[1] >> 16) & 1) && os_avx512;
        p->avx512_vnni = ((r[2] >> 11) & 1) && os_avx512;
        __cpuidex(r, 7, 1);
        p->avx_vnni = ((r[0] >> 4) & 1) && os_avx;
    }
    DWORD len = 0; GetLogicalProcessorInformation(NULL, &len);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf = malloc(len);
    if (buf && GetLogicalProcessorInformation(buf, &len))
        for (DWORD i = 0; i < len / sizeof *buf; i++)
            if (buf[i].Relationship == RelationProcessorCore) {
                p->cores++;
                for (ULONG_PTR m = buf[i].ProcessorMask; m; m &= m - 1) p->threads++;
            }
    free(buf);
}

/* ---------------- NVIDIA via the driver API (nvcuda.dll), loaded at run time ---------------- */
typedef int (__stdcall *cu_i)(unsigned);
typedef int (__stdcall *cu_pi)(int *);
typedef int (__stdcall *cu_get)(int *, int);
typedef int (__stdcall *cu_name)(char *, int, int);
typedef int (__stdcall *cu_mem)(size_t *, int);
typedef int (__stdcall *cu_attr)(int *, int, int);
typedef int (__stdcall *cu_ctx)(void **, unsigned, int);
typedef int (__stdcall *cu_ctxd)(void *);
typedef int (__stdcall *cu_info)(size_t *, size_t *);
typedef int (__stdcall *cu_hostalloc)(void **, size_t, unsigned);
typedef int (__stdcall *cu_hostfree)(void *);
typedef int (__stdcall *cu_alloc)(unsigned long long *, size_t);
typedef int (__stdcall *cu_free)(unsigned long long);
typedef int (__stdcall *cu_h2d)(unsigned long long, const void *, size_t);

/* Pinned host -> VRAM copy speed: 64 MB, 1 warm-up, best of 3 (synchronous copies). 0 when anything fails. */
static double probe_h2d(HMODULE h) {
    cu_hostalloc ha = (cu_hostalloc)GetProcAddress(h, "cuMemHostAlloc");
    cu_hostfree hf = (cu_hostfree)GetProcAddress(h, "cuMemFreeHost");
    cu_alloc da = (cu_alloc)GetProcAddress(h, "cuMemAlloc_v2");
    cu_free df = (cu_free)GetProcAddress(h, "cuMemFree_v2");
    cu_h2d cp = (cu_h2d)GetProcAddress(h, "cuMemcpyHtoD_v2");
    const size_t n = 64u << 20;
    void *host = NULL; unsigned long long dev = 0; double best = 0;
    if (!ha || !hf || !da || !df || !cp || ha(&host, n, 0)) return 0;
    if (!da(&dev, n)) {
        memset(host, 1, n);
        LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f);
        for (int i = 0; i < 4; i++) {
            QueryPerformanceCounter(&a);
            if (cp(dev, host, n)) { best = 0; break; }
            QueryPerformanceCounter(&b);
            double gbs = n / ((double)(b.QuadPart - a.QuadPart) / f.QuadPart) / 1e9;
            if (i > 0 && gbs > best) best = gbs;
        }
        df(dev);
    }
    hf(host);
    return best;
}

static void probe_cuda(rope_probe *p) {
    /* ROPE_NVCUDA_DLL overrides the driver library name: diagnostics, and tests of a machine with no driver. */
    const char *dll = getenv("ROPE_NVCUDA_DLL");
    HMODULE h = LoadLibraryA(dll && *dll ? dll : "nvcuda.dll");
    if (!h) return;
    cu_i init = (cu_i)GetProcAddress(h, "cuInit");
    cu_pi count = (cu_pi)GetProcAddress(h, "cuDeviceGetCount"), drv = (cu_pi)GetProcAddress(h, "cuDriverGetVersion");
    cu_get get = (cu_get)GetProcAddress(h, "cuDeviceGet");
    cu_name name = (cu_name)GetProcAddress(h, "cuDeviceGetName");
    cu_mem mem = (cu_mem)GetProcAddress(h, "cuDeviceTotalMem_v2");
    cu_attr attr = (cu_attr)GetProcAddress(h, "cuDeviceGetAttribute");
    cu_ctx ctx = (cu_ctx)GetProcAddress(h, "cuCtxCreate_v2");
    cu_ctxd ctxd = (cu_ctxd)GetProcAddress(h, "cuCtxDestroy_v2");
    cu_info info = (cu_info)GetProcAddress(h, "cuMemGetInfo_v2");
    int n = 0, dev = 0;
    if (!init || !count || !get || !name || !mem || !attr || init(0) || count(&n) || n < 1 || get(&dev, 0)) { FreeLibrary(h); return; }
    size_t total = 0;
    name(p->cuda_name, sizeof p->cuda_name, dev);
    mem(&total, dev);
    attr(&p->cc_major, 75, dev);     /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
    attr(&p->cc_minor, 76, dev);     /* ..._MINOR */
    attr(&p->sm_count, 16, dev);     /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */
    attr(&p->clock_khz, 13, dev);    /* CU_DEVICE_ATTRIBUTE_CLOCK_RATE (boost clock, kHz) */
    if (drv) drv(&p->cuda_driver);
    p->vram_total = total;
    void *c = NULL;
    if (ctx && ctxd && info && !ctx(&c, 0, dev)) {
        size_t f = 0, t = 0;
        if (!info(&f, &t)) p->vram_free = f;          /* measured before the copy test allocates */
        p->h2d_gbs = probe_h2d(h);
        ctxd(c);
    }
    p->cuda_ok = 1;
    /* nvcuda.dll stays loaded: unloading a driver library after cuInit is not supported */
}

/* ---------------- all adapters via DXGI ---------------- */
static void probe_dxgi(rope_probe *p) {
    IDXGIFactory1 *f = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory1, (void **)&f))) return;
    IDXGIAdapter1 *a;
    for (UINT i = 0; p->n_adapters < ROPE_MAX_GPUS && IDXGIFactory1_EnumAdapters1(f, i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 d;
        if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            rope_adapter *r = &p->adapters[p->n_adapters++];
            WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, r->name, sizeof r->name, NULL, NULL);
            r->vendor = (int)d.VendorId;
            r->dedicated_bytes = d.DedicatedVideoMemory;
            r->shared_bytes = d.SharedSystemMemory;
            r->integrated = d.DedicatedVideoMemory < (512ull << 20);
        }
        IDXGIAdapter1_Release(a);
    }
    IDXGIFactory1_Release(f);
}

/* ---------------- storage: free space, SSD/HDD, bus, sequential speed with the cache bypassed ---------------- */
static void probe_disk_type(rope_probe *p, const wchar_t *wdir) {
    wchar_t vol[MAX_PATH], dev[64];
    p->disk_ssd = -1; strcpy(p->disk_bus, "unknown");
    if (!GetVolumePathNameW(wdir, vol, MAX_PATH) || wcslen(vol) < 2 || vol[1] != L':') return;
    swprintf(dev, 64, L"\\\\.\\%c:", vol[0]);
    HANDLE h = CreateFileW(dev, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);   /* 0 access: no admin needed */
    if (h == INVALID_HANDLE_VALUE) return;
    STORAGE_PROPERTY_QUERY q = {0}; DWORD got;
    q.PropertyId = StorageDeviceSeekPenaltyProperty; q.QueryType = PropertyStandardQuery;
    DEVICE_SEEK_PENALTY_DESCRIPTOR sp;
    if (DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof q, &sp, sizeof sp, &got, NULL)) p->disk_ssd = !sp.IncursSeekPenalty;
    q.PropertyId = StorageDeviceProperty;
    STORAGE_DEVICE_DESCRIPTOR dd;
    if (DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof q, &dd, sizeof dd, &got, NULL)) {
        static const char *B[] = {"unknown", "SCSI", "ATAPI", "ATA", "1394", "SSA", "FC", "USB", "RAID", "iSCSI", "SAS", "SATA",
                                  "SD", "MMC", "Virtual", "FileBackedVirtual", "Spaces", "NVMe", "SCM", "UFS"};
        if ((unsigned)dd.BusType < sizeof B / sizeof B[0]) snprintf(p->disk_bus, sizeof p->disk_bus, "%s", B[dd.BusType]);
    }
    CloseHandle(h);
}

/* Write a temp file with write-through, then read it back unbuffered with 4 overlapped 4 MiB requests. */
static void probe_disk_speed(rope_probe *p, const wchar_t *wdir, int mb) {
    const DWORD BLK = 4u << 20; const int QD = 4;
    wchar_t path[MAX_PATH];
    swprintf(path, MAX_PATH, L"%ls\\rope_probe_%lu.tmp", wdir, GetCurrentProcessId());
    uint8_t *buf = VirtualAlloc(NULL, (SIZE_T)BLK * QD, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);  /* page-aligned */
    if (!buf) return;
    for (DWORD i = 0; i < BLK * QD; i++) buf[i] = (uint8_t)(i * 2654435761u >> 24);   /* incompressible-ish */
    int nblk = mb / 4; LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f);
    HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, NULL);
    if (h == INVALID_HANDLE_VALUE) { VirtualFree(buf, 0, MEM_RELEASE); return; }
    QueryPerformanceCounter(&t0);
    int ok = 1; DWORD w;
    for (int i = 0; i < nblk && ok; i++) ok = WriteFile(h, buf + (size_t)(i % QD) * BLK, BLK, &w, NULL) && w == BLK;
    QueryPerformanceCounter(&t1);
    CloseHandle(h);
    if (ok) p->disk_write_gbs = (double)nblk * BLK / 1e9 / ((double)(t1.QuadPart - t0.QuadPart) / f.QuadPart);
    h = CreateFileW(path, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (ok && h != INVALID_HANDLE_VALUE) {
        OVERLAPPED ov[4]; HANDLE ev[4]; int next = 0, done = 0;
        for (int q = 0; q < QD; q++) { memset(&ov[q], 0, sizeof ov[q]); ev[q] = CreateEventW(NULL, TRUE, FALSE, NULL); ov[q].hEvent = ev[q]; }
        QueryPerformanceCounter(&t0);
        for (int q = 0; q < QD && next < nblk; q++, next++) {
            ov[q].Offset = (DWORD)((uint64_t)next * BLK); ov[q].OffsetHigh = (DWORD)(((uint64_t)next * BLK) >> 32);
            if (!ReadFile(h, buf + (size_t)q * BLK, BLK, NULL, &ov[q]) && GetLastError() != ERROR_IO_PENDING) ok = 0;
        }
        while (ok && done < nblk) {
            for (int q = 0; q < QD && done < nblk; q++) {
                DWORD got;
                if (!GetOverlappedResult(h, &ov[q], &got, TRUE)) { ok = 0; break; }
                done++;
                if (next < nblk) {
                    ResetEvent(ev[q]);
                    ov[q].Offset = (DWORD)((uint64_t)next * BLK); ov[q].OffsetHigh = (DWORD)(((uint64_t)next * BLK) >> 32); next++;
                    if (!ReadFile(h, buf + (size_t)q * BLK, BLK, NULL, &ov[q]) && GetLastError() != ERROR_IO_PENDING) ok = 0;
                }
            }
        }
        QueryPerformanceCounter(&t1);
        if (ok) p->disk_read_gbs = (double)nblk * BLK / 1e9 / ((double)(t1.QuadPart - t0.QuadPart) / f.QuadPart);
        for (int q = 0; q < QD; q++) CloseHandle(ev[q]);
        CloseHandle(h);
    }
    DeleteFileW(path);
    VirtualFree(buf, 0, MEM_RELEASE);
}

/* One string from the registry (read only). */
static void reg_str(HKEY root, const char *key, const char *name, char *out, size_t cap) {
    DWORD n = (DWORD)cap;
    if (RegGetValueA(root, key, name, RRF_RT_REG_SZ, NULL, out, &n) != ERROR_SUCCESS) out[0] = 0;
}

/* Windows edition, machine maker and model, CPU clock, power source. No serial numbers, no user or computer name. */
static void probe_system(rope_probe *p) {
    char prod[64] = "", disp[32] = "", maker[48] = "", model[48] = "";
    reg_str(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "ProductName", prod, sizeof prod);
    reg_str(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "DisplayVersion", disp, sizeof disp);
    snprintf(p->os_name, sizeof p->os_name, "%s %s", prod, disp);
    reg_str(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\BIOS", "SystemManufacturer", maker, sizeof maker);
    reg_str(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\BIOS", "SystemProductName", model, sizeof model);
    snprintf(p->machine, sizeof p->machine, "%s %s", maker, model);
    DWORD mhz = 0, n = sizeof mhz;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "~MHz", RRF_RT_REG_DWORD, NULL, &mhz, &n) == ERROR_SUCCESS) p->cpu_mhz = (int)mhz;
    p->on_ac = -1; p->battery_pct = -1; p->battery_saver = -1;
    SYSTEM_POWER_STATUS sp;
    if (GetSystemPowerStatus(&sp)) {
        if (!(sp.BatteryFlag & 128) && sp.BatteryFlag != 255) {          /* a battery exists */
            p->on_ac = sp.ACLineStatus == 1 ? 1 : sp.ACLineStatus == 0 ? 0 : -1;
            if (sp.BatteryLifePercent <= 100) p->battery_pct = sp.BatteryLifePercent;
        }
        p->battery_saver = (sp.SystemStatusFlag & 1) ? 1 : 0;
    }
}

/* The report must not carry the user's name: of a path only the drive stays. */
static void drive_only(const wchar_t *full, char *out, size_t cap) {
    if (full[0] && full[1] == L':') snprintf(out, cap, "%c:\\", (char)full[0]);
    else snprintf(out, cap, "network or other drive");
}

int rope_probe_run(rope_probe *p, const char *dir, int disk_test_mb) {
    memset(p, 0, sizeof *p);
    probe_system(p);
    OSVERSIONINFOEXW v = {sizeof v};
    typedef LONG (WINAPI *rtl_gv)(OSVERSIONINFOEXW *);
    rtl_gv gv = (rtl_gv)GetProcAddress(GetModuleHandleA("ntdll.dll"), "RtlGetVersion");
    if (gv && !gv(&v)) snprintf(p->os, sizeof p->os, "Windows %lu.%lu.%lu", v.dwMajorVersion, v.dwMinorVersion, v.dwBuildNumber);
    MEMORYSTATUSEX ms = {sizeof ms};
    if (GlobalMemoryStatusEx(&ms)) { p->ram_total = ms.ullTotalPhys; p->ram_free = ms.ullAvailPhys; }
    ULONGLONG kb = 0;
    if (GetPhysicallyInstalledSystemMemory(&kb)) p->ram_installed = kb * 1024;
    probe_cpu(p);
    probe_cuda(p);
    probe_dxgi(p);
    wchar_t wdir[MAX_PATH], full[MAX_PATH];
    if (!MultiByteToWideChar(CP_UTF8, 0, dir && *dir ? dir : ".", -1, wdir, MAX_PATH)) return -1;
    if (!GetFullPathNameW(wdir, MAX_PATH, full, NULL)) return -1;
    drive_only(full, p->disk_dir, sizeof p->disk_dir);
    ULARGE_INTEGER fr, tot;
    if (GetDiskFreeSpaceExW(full, &fr, &tot, NULL)) { p->disk_free = fr.QuadPart; p->disk_total = tot.QuadPart; }
    probe_disk_type(p, full);
    if (disk_test_mb >= 64) probe_disk_speed(p, full, disk_test_mb);
    return 0;
}
#elif !defined(__linux__)
int rope_probe_run(rope_probe *p, const char *dir, int disk_test_mb) { (void)dir; (void)disk_test_mb; memset(p, 0, sizeof *p); return -1; }
#endif

/* JSON is built into a growable buffer, so it can be printed or handed to Rust. */
static char *g_jb; static size_t g_jn, g_jcap;
static void jgrow(size_t add) {
    if (g_jn + add + 1 <= g_jcap) return;
    size_t cap = g_jcap ? g_jcap : 4096;
    while (cap < g_jn + add + 1) cap *= 2;
    char *nb = realloc(g_jb, cap);
    if (!nb) abort();
    g_jb = nb; g_jcap = cap;
}
static void jp(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap); va_end(ap);
    if (n < 0) return;
    jgrow((size_t)n);
    va_start(ap, fmt); vsnprintf(g_jb + g_jn, (size_t)n + 1, fmt, ap); va_end(ap);
    g_jn += (size_t)n;
}
static void jc(char c) { jgrow(1); g_jb[g_jn++] = c; g_jb[g_jn] = 0; }

static void jstr(const char *k, const char *v, int comma) {
    jp("  \"%s\": \"", k);
    for (; *v; v++) { if (*v == '"' || *v == '\\') jc('\\'); jc(*v); }
    jp("\"%s\n", comma ? "," : "");
}

static void build_json(const rope_probe *p);
static char g_pct[16];

void rope_probe_print_json(const rope_probe *p) { build_json(p); fputs(g_jb, stdout); }

/* Run the probe and copy its JSON into out (NUL-terminated). Returns the length, or -1 if it does not fit / fails. */
int rope_c_probe_json(const char *dir, int disk_mb, char *out, size_t cap) {
    rope_probe p;
    if (rope_probe_run(&p, dir, disk_mb)) return -1;
    build_json(&p);
    if (g_jn + 1 > cap) return -1;
    memcpy(out, g_jb, g_jn + 1);
    return (int)g_jn;
}

static void build_json(const rope_probe *p) {
    const double G = 1024.0 * 1024 * 1024;
    g_jn = 0; jgrow(0); g_jb[0] = 0;
    jp("{\n");
    jstr("os", p->os, 1);
    jstr("os_name", p->os_name, 1);
    jstr("machine", p->machine, 1);
    jp("  \"cpu_mhz\": %d,\n", p->cpu_mhz);
    jp("  \"power\": {\"on_ac\": %s, \"battery_percent\": %s, \"battery_saver\": %s},\n",
       p->on_ac < 0 ? "null" : p->on_ac ? "true" : "false",
       p->battery_pct < 0 ? "null" : (snprintf(g_pct, sizeof g_pct, "%d", p->battery_pct), g_pct),
       p->battery_saver < 0 ? "null" : p->battery_saver ? "true" : "false");
    jp("  \"ram_installed_gib\": %.2f,\n  \"ram_usable_gib\": %.2f,\n  \"ram_free_gib\": %.2f,\n",
           p->ram_installed / G, p->ram_total / G, p->ram_free / G);
    jstr("cpu", p->cpu_name, 1);
    jp("  \"cores\": %d,\n  \"threads\": %d,\n", p->cores, p->threads);
    jp("  \"cpu_path\": \"%s\",\n", rope_cpu_isa_name());          /* the kernel variant this process runs (P2.2.5) */
    jp("  \"cpu_flags\": {\"sse42\": %s, \"avx\": %s, \"avx2\": %s, \"fma\": %s, \"f16c\": %s, \"avx512f\": %s, \"avx512_vnni\": %s, \"avx_vnni\": %s},\n",
           p->sse42 ? "true" : "false", p->avx ? "true" : "false", p->avx2 ? "true" : "false", p->fma ? "true" : "false",
           p->f16c ? "true" : "false", p->avx512f ? "true" : "false", p->avx512_vnni ? "true" : "false", p->avx_vnni ? "true" : "false");
    if (p->cuda_ok) {
        jp("  \"gpu\": {\"api\": \"cuda\", \"name\": \"%s\", \"vram_total_gib\": %.2f, \"vram_free_gib\": %.2f, \"compute_capability\": \"%d.%d\", "
               "\"sm\": \"sm_%d%d\", \"multiprocessors\": %d, \"clock_mhz\": %d, \"h2d_gbs\": %.2f, \"driver_cuda_version\": \"%d.%d\", \"tensor_cores\": %s},\n",
               p->cuda_name, p->vram_total / G, p->vram_free / G, p->cc_major, p->cc_minor, p->cc_major, p->cc_minor, p->sm_count,
               p->clock_khz / 1000, p->h2d_gbs,
               p->cuda_driver / 1000, (p->cuda_driver % 1000) / 10,
               /* tensor cores: Volta+ except the Turing GTX 16xx line (TU116/TU117 have none) */
               (p->cc_major > 7 || (p->cc_major == 7 && p->cc_minor == 0) || (p->cc_major == 7 && p->cc_minor == 5 && !strstr(p->cuda_name, "GTX 16"))) ? "true" : "false");
    } else {
        jp("  \"gpu\": null,\n");
    }
    jp("  \"adapters\": [");
    for (int i = 0; i < p->n_adapters; i++) {
        const rope_adapter *a = &p->adapters[i];
        jp("%s\n    {\"name\": \"%s\", \"vendor\": \"0x%04X\", \"dedicated_gib\": %.2f, \"shared_gib\": %.2f, \"integrated\": %s}",
               i ? "," : "", a->name, a->vendor, a->dedicated_bytes / G, a->shared_bytes / G, a->integrated ? "true" : "false");
    }
    jp("%s],\n", p->n_adapters ? "\n  " : "");
    jp("  \"disk\": {\"dir\": \"");
    for (const char *s = p->disk_dir; *s; s++) { if (*s == '"' || *s == '\\') jc('\\'); jc(*s); }
    jp("\", \"free_gib\": %.2f, \"total_gib\": %.2f, \"ssd\": %s, \"bus\": \"%s\", \"seq_read_gbs\": %.2f, \"seq_write_gbs\": %.2f}\n",
           p->disk_free / G, p->disk_total / G, p->disk_ssd < 0 ? "null" : p->disk_ssd ? "true" : "false", p->disk_bus,
           p->disk_read_gbs, p->disk_write_gbs);
    jp("}\n");
}
