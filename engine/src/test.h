/* Tiny unit-test runner: tests register in test.c's table; `rope test [filter]` runs them. */
#pragma once
#include <stdio.h>

extern int rope_test_failures;

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); rope_test_failures++; return; } } while (0)

int rope_run_tests(const char *filter);
