/* Configuration file defining macros needed by the PET library */

/* Define to Diagnostic for newer versions of clang */
#cmakedefine DiagnosticInfo Diagnostic

/* Return type of HandleTopLevelDeclReturn */
#cmakedefine HandleTopLevelDeclReturn bool

/* Return value of HandleTopLevelDeclContinue */
#cmakedefine HandleTopLevelDeclContinue true

/* Define if TargetInfo::CreateTargetInfo takes shared_ptr */
#cmakedefine CREATETARGETINFO_TAKES_SHARED_PTR 1

/* Defined if CompilerInstance::setInvocation takes a shared_ptr */
#cmakedefine SETINVOCATION_TAKES_SHARED_PTR 1

/* Define if HeaderSearchOptions::AddPath takes 4 arguments */
#cmakedefine ADDPATH_TAKES_4_ARGUMENTS 1

/* Define to ext_implicit_function_decl for older versions of clang */
#cmakedefine ext_implicit_function_decl_c99 ext_implicit_function_decl

/* Define if SourceManager has translateLineCol method */
#cmakedefine HAVE_TRANSLATELINECOL 1

/* Define to 1 if you have the declaration of `ffs' */
#cmakedefine HAVE_DECL_FFS 1

/* Define to 1 if you have the declaration of `strcasecmp' */
#cmakedefine HAVE_DECL_STRCASECMP 1

/* Define to 1 if you have the declaration of `strncasecmp' */
#cmakedefine HAVE_DECL_STRNCASECMP 1

/* Define to 1 if you have the `getrusage' function */
#cmakedefine HAVE_GETRUSAGE 1

/* Define to 1 if you have the `gettimeofday' function */
#cmakedefine HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have the <ssize_t> type */
#cmakedefine HAVE_SSIZE_T 1

/* Package information */
#define PACKAGE "pet"
#define PACKAGE_BUGREPORT ""
#define PACKAGE_NAME "pet"
#define PACKAGE_STRING "pet @PET_VERSION@"
#define PACKAGE_TARNAME "pet"
#define PACKAGE_URL ""
#define PACKAGE_VERSION "@PET_VERSION@"
#define VERSION "@PET_VERSION@"

/* Define to 1 if NO_CLANG is enabled */
#cmakedefine NO_CLANG 1

/* Define to 1 if you have the <stdint.h> header file */
#cmakedefine HAVE_STDINT_H 1

/* Define if ArraySizeModifier appears inside ArrayType */
#cmakedefine USE_NESTED_ARRAY_SIZE_MODIFIER 1

/* Define if getBeginLoc and getEndLoc should be used */
#cmakedefine HAVE_BEGIN_END_LOC 1

/* Define to getFileLocWithOffset for older versions of clang */
#cmakedefine getFileLocWithOffset getLocWithOffset

/* Define if getTypeInfo returns TypeInfo object */
#cmakedefine GETTYPEINFORETURNSTYPEINFO 1

/* Define if clang/Basic/DiagnosticOptions.h exists */
#cmakedefine HAVE_BASIC_DIAGNOSTICOPTIONS_H 1

/* Define if clang/Lex/HeaderSearchOptions.h exists */
#cmakedefine HAVE_LEX_HEADERSEARCHOPTIONS_H 1

/* Define if clang/Basic/LangStandard.h exists */
#cmakedefine HAVE_CLANG_BASIC_LANGSTANDARD_H 1

/* Define if clang/Lex/PreprocessorOptions.h exists */
#cmakedefine HAVE_LEX_PREPROCESSOROPTIONS_H 1

/* Define if llvm/Option/Arg.h exists */
#cmakedefine HAVE_LLVM_OPTION_ARG_H 1

/* Define if CompilerInstance::createPreprocessor takes TranslationUnitKind */
#cmakedefine CREATEPREPROCESSOR_TAKES_TUKIND 1

/* Define if SourceManager has a setMainFileID method */
#cmakedefine HAVE_SETMAINFILEID 1

/* Define if CompilerInvocation::setLangDefaults takes 5 arguments */
#cmakedefine SETLANGDEFAULTS_TAKES_5_ARGUMENTS 1

/* Define to class with setLangDefaults method */
#cmakedefine SETLANGDEFAULTS CompilerInvocation

/* Define to Language::C or InputKind::C for newer versions of clang */
#cmakedefine IK_C Language::C

/* Define compatibility macros for getLocStart/getLocEnd vs getBeginLoc/getEndLoc */
#ifndef HAVE_BEGIN_END_LOC
#define getBeginLoc getLocStart
#define getEndLoc getLocEnd
#endif

/* Define compatibility macros for ArraySizeModifier */
#ifndef USE_NESTED_ARRAY_SIZE_MODIFIER
enum ArraySizeModifier { Normal, Static, Star };
#endif