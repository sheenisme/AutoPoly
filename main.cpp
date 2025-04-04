#include <iostream>
#include <string>
#include <cstdio>

// 包含PPCG头文件
extern "C" {
#include "ppcg.h"
#include "ppcg_options.h"

extern int ppcg_main(int argc, char *argv[]);
}

int main(int argc, char *argv[]) {
    ppcg_main(argc, argv);
    return 0;
}