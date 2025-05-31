#ifndef __PPCG_WRAPPER_H__
#define __PPCG_WRAPPER_H__

// This header file is a wrapper for the PPCG
extern "C" {
#include "ppcg.h"
#include "ppcg_options.h"

extern int ppcg_main(int argc, char *argv[]);
}

#endif // __PPCG_WRAPPER_H__