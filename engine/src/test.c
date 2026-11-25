#include "test.h"
#include <string.h>

int rope_test_failures = 0;

#include "kernels.h"
uint16_t rope_f2h_avx2(float f);
uint16_t rope_f2h_sse41(float f);

/* P2.2.5: a valid CPU path is chosen; the SSE4.1 path's software float->half equals F16C bit for bit (when the CPU
   has F16C) on normal, subnormal, rounding-tie, overflow, infinity and NaN inputs. */
static void test_cpu_dispatch(void) {
    int isa = rope_cpu_isa();
    CHECK(isa == 1 || isa == 2);
    if (isa != 2) return;
    int bad = 0;
    for (uint32_t i = 0; i < 4000000; i++) {
        uint32_t bits = i * 1073u + (i >> 7) * 2654435761u;          /* spread over all exponents and signs */
        float f; memcpy(&f, &bits, 4);
        if (f != f) continue;                                        /* NaN payloads may differ: checked below */
        bad += rope_f2h_avx2(f) != rope_f2h_sse41(f);
    }
    const float edge[] = { 0.0f, -0.0f, 65504.0f, 65520.0f, 1e9f, -1e9f, 6.1e-5f, 5.96e-8f, 2.98e-8f, 1.0f + 1.0f / 2048, 1.0f + 3.0f / 2048 };
    for (size_t i = 0; i < sizeof edge / sizeof edge[0]; i++) bad += rope_f2h_avx2(edge[i]) != rope_f2h_sse41(edge[i]);
    float nan; uint32_t nb = 0x7FC00000u; memcpy(&nan, &nb, 4);
    CHECK((rope_f2h_sse41(nan) & 0x7C00) == 0x7C00 && (rope_f2h_sse41(nan) & 0x3FF) != 0);
    CHECK(bad == 0);
}

#ifdef ROPE_CUDA
int rope_cuda_device_count(void);
static void test_cuda_device(void) { CHECK(rope_cuda_device_count() >= 1); }
void rope_register_cuda_tests(void (*reg)(const char *, void (*)(void)));
#endif

/* Each test file registers its tests here, so adding a file needs one line. */
void rope_register_kernel_tests(void (*reg)(const char *, void (*)(void)));

static struct { const char *name; void (*fn)(void); } tests[256];
static int n_tests;
static void reg(const char *name, void (*fn)(void)) { tests[n_tests].name = name; tests[n_tests++].fn = fn; }

int rope_run_tests(const char *filter) {
    n_tests = 0;
    reg("cpu_dispatch", test_cpu_dispatch);
#ifdef ROPE_CUDA
    reg("cuda_device", test_cuda_device);
    rope_register_cuda_tests(reg);
#endif
    rope_register_kernel_tests(reg);
    int run = 0;
    for (int i = 0; i < n_tests; i++) {
        if (filter && !strstr(tests[i].name, filter)) continue;
        int before = rope_test_failures;
        tests[i].fn();
        printf("%s %s\n", rope_test_failures == before ? "  ok  " : "  FAIL", tests[i].name);
        run++;
    }
    printf("%d tests, %d failed\n", run, rope_test_failures);
    return rope_test_failures ? 1 : 0;
}
