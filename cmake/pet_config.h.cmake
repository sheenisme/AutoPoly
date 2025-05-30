/* pet config.h.  Generated from CMake configuration. */

/* Define if HeaderSearchOptions::AddPath takes 4 arguments */
#cmakedefine ADDPATH_TAKES_4_ARGUMENTS @ADDPATH_TAKES_4_ARGUMENTS@

/* Clang installation prefix */
#cmakedefine CLANG_PREFIX "@CLANG_PREFIX@"

/* Define if CompilerInstance::createDiagnostics takes argc and argv */
#cmakedefine CREATEDIAGNOSTICS_TAKES_ARG @CREATEDIAGNOSTICS_TAKES_ARG@

/* Define if CompilerInstance::createPreprocessor takes TranslationUnitKind */
#cmakedefine CREATEPREPROCESSOR_TAKES_TUKIND @CREATEPREPROCESSOR_TAKES_TUKIND@

/* Define if TargetInfo::CreateTargetInfo takes pointer */
#cmakedefine CREATETARGETINFO_TAKES_POINTER @CREATETARGETINFO_TAKES_POINTER@

/* Define if TargetInfo::CreateTargetInfo takes shared_ptr */
#cmakedefine CREATETARGETINFO_TAKES_SHARED_PTR @CREATETARGETINFO_TAKES_SHARED_PTR@

/* Define if CompilerInvocation::CreateFromArgs takes ArrayRef */
#cmakedefine CREATE_FROM_ARGS_TAKES_ARRAYREF @CREATE_FROM_ARGS_TAKES_ARRAYREF@

/* Define if Driver constructor takes default image name */
#cmakedefine DRIVER_CTOR_TAKES_DEFAULTIMAGENAME @DRIVER_CTOR_TAKES_DEFAULTIMAGENAME@

/* Define to DiagnosticClient for older versions of clang */
#cmakedefine DiagnosticConsumer @DiagnosticConsumer@

/* Define to Diagnostic for newer versions of clang */
#cmakedefine DiagnosticInfo @DiagnosticInfo@

/* Define to Diagnostic for older versions of clang */
#cmakedefine DiagnosticsEngine @DiagnosticsEngine@

/* Define if getTypeInfo returns TypeInfo object */
#cmakedefine GETTYPEINFORETURNSTYPEINFO @GETTYPEINFORETURNSTYPEINFO@

/* Define if llvm/ADT/OwningPtr.h exists */
#cmakedefine HAVE_ADT_OWNINGPTR_H @HAVE_ADT_OWNINGPTR_H@

/* Define if clang/Basic/DiagnosticOptions.h exists */
#cmakedefine HAVE_BASIC_DIAGNOSTICOPTIONS_H @HAVE_BASIC_DIAGNOSTICOPTIONS_H@

/* Define if getBeginLoc and getEndLoc should be used */
#cmakedefine HAVE_BEGIN_END_LOC @HAVE_BEGIN_END_LOC@

/* Define if clang/Basic/LangStandard.h exists */
#cmakedefine HAVE_CLANG_BASIC_LANGSTANDARD_H @HAVE_CLANG_BASIC_LANGSTANDARD_H@

/* Define if Driver constructor takes CXXIsProduction argument */
#cmakedefine HAVE_CXXISPRODUCTION @HAVE_CXXISPRODUCTION@

/* Define if DecayedType is defined */
#cmakedefine HAVE_DECAYEDTYPE @HAVE_DECAYEDTYPE@

/* Define to 1 if you have the <dlfcn.h> header file. */
#cmakedefine HAVE_DLFCN_H @HAVE_DLFCN_H@

/* Define if SourceManager has findLocationAfterToken method */
#cmakedefine HAVE_FINDLOCATIONAFTERTOKEN @HAVE_FINDLOCATIONAFTERTOKEN@

/* Define to 1 if you have the <inttypes.h> header file. */
#cmakedefine HAVE_INTTYPES_H @HAVE_INTTYPES_H@

/* Define if Driver constructor takes IsProduction argument */
#cmakedefine HAVE_ISPRODUCTION @HAVE_ISPRODUCTION@

/* Define if clang/Lex/HeaderSearchOptions.h exists */
#cmakedefine HAVE_LEX_HEADERSEARCHOPTIONS_H @HAVE_LEX_HEADERSEARCHOPTIONS_H@

/* Define if clang/Lex/PreprocessorOptions.h exists */
#cmakedefine HAVE_LEX_PREPROCESSOROPTIONS_H @HAVE_LEX_PREPROCESSOROPTIONS_H@

/* Define if llvm/Option/Arg.h exists */
#cmakedefine HAVE_LLVM_OPTION_ARG_H @HAVE_LLVM_OPTION_ARG_H@

/* Define if SourceManager has a setMainFileID method */
#cmakedefine HAVE_SETMAINFILEID @HAVE_SETMAINFILEID@

/* Define if DiagnosticsEngine::setDiagnosticGroupWarningAsError is available */
#cmakedefine HAVE_SET_DIAGNOSTIC_GROUP_WARNING_AS_ERROR @HAVE_SET_DIAGNOSTIC_GROUP_WARNING_AS_ERROR@

/* Define to 1 if you have the <stdint.h> header file. */
#cmakedefine HAVE_STDINT_H @HAVE_STDINT_H@

/* Define to 1 if you have the <stdio.h> header file. */
#cmakedefine HAVE_STDIO_H @HAVE_STDIO_H@

/* Define to 1 if you have the <stdlib.h> header file. */
#cmakedefine HAVE_STDLIB_H @HAVE_STDLIB_H@

/* Define if StmtRange class is available */
#cmakedefine HAVE_STMTRANGE @HAVE_STMTRANGE@

/* Define to 1 if you have the <strings.h> header file. */
#cmakedefine HAVE_STRINGS_H @HAVE_STRINGS_H@

/* Define to 1 if you have the <string.h> header file. */
#cmakedefine HAVE_STRING_H @HAVE_STRING_H@

/* Define to 1 if you have the <sys/stat.h> header file. */
#cmakedefine HAVE_SYS_STAT_H @HAVE_SYS_STAT_H@

/* Define to 1 if you have the <sys/types.h> header file. */
#cmakedefine HAVE_SYS_TYPES_H @HAVE_SYS_TYPES_H@

/* Define if llvm/TargetParser/Host.h exists */
#cmakedefine HAVE_TARGETPARSER_HOST_H @HAVE_TARGETPARSER_HOST_H@

/* Define if SourceManager has translateLineCol method */
#cmakedefine HAVE_TRANSLATELINECOL @HAVE_TRANSLATELINECOL@

/* Define to 1 if you have the <unistd.h> header file. */
#cmakedefine HAVE_UNISTD_H @HAVE_UNISTD_H@

/* Return type of HandleTopLevelDeclReturn */
#cmakedefine HandleTopLevelDeclContinue @HandleTopLevelDeclContinue@

/* Return type of HandleTopLevelDeclReturn */
#cmakedefine HandleTopLevelDeclReturn @HandleTopLevelDeclReturn@

/* Define to Language::C or InputKind::C for newer versions of clang */
#cmakedefine IK_C @IK_C@

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "pet"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "isl-development@googlegroups.com"

/* Define to the full name of this package. */
#define PACKAGE_NAME "pet"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "pet @PET_VERSION@"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "pet"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "@PET_VERSION@"

/* Define to PragmaIntroducerKind for older versions of clang */
#cmakedefine PragmaIntroducer @PragmaIntroducer@

/* Defined if CompilerInstance::setInvocation takes a shared_ptr */
#cmakedefine SETINVOCATION_TAKES_SHARED_PTR @SETINVOCATION_TAKES_SHARED_PTR@

/* Define to class with setLangDefaults method */
#cmakedefine SETLANGDEFAULTS @SETLANGDEFAULTS@

/* Define if CompilerInvocation::setLangDefaults takes 5 arguments */
#cmakedefine SETLANGDEFAULTS_TAKES_5_ARGUMENTS @SETLANGDEFAULTS_TAKES_5_ARGUMENTS@

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#cmakedefine STDC_HEADERS @STDC_HEADERS@

/* Define to TypedefDecl for older versions of clang */
#cmakedefine TypedefNameDecl @TypedefNameDecl@

/* Define if Driver::BuildCompilation takes ArrayRef */
#cmakedefine USE_ARRAYREF @USE_ARRAYREF@

/* Define if ArraySizeModifier appears inside ArrayType */
#cmakedefine USE_NESTED_ARRAY_SIZE_MODIFIER @USE_NESTED_ARRAY_SIZE_MODIFIER@

/* Version number of package */
#define VERSION "@PET_VERSION@"

/* Define to ext_implicit_function_decl for older versions of clang */
#cmakedefine ext_implicit_function_decl_c99 @ext_implicit_function_decl_c99@

/* Define to getHostTriple for older versions of clang */
#cmakedefine getDefaultTargetTriple @getDefaultTargetTriple@

/* Define to getInstantiationColumnNumber for older versions of clang */
#cmakedefine getExpansionColumnNumber @getExpansionColumnNumber@

/* Define to getInstantiationLineNumber for older versions of clang */
#cmakedefine getExpansionLineNumber @getExpansionLineNumber@

/* Define to getInstantiationLoc for older versions of clang */
#cmakedefine getExpansionLoc @getExpansionLoc@

/* Define to getLangOptions for older versions of clang */
#cmakedefine getLangOpts @getLangOpts@

/* Define to getFileLocWithOffset for older versions of clang */
#cmakedefine getLocWithOffset @getLocWithOffset@

/* Define to getResultType for older versions of clang */
#cmakedefine getReturnType @getReturnType@

/* Define to getTypedefForAnonDecl for older versions of clang */
#cmakedefine getTypedefNameForAnonDecl @getTypedefNameForAnonDecl@

/* Define to InitializeBuiltins for older versions of clang */
#cmakedefine initializeBuiltins @initializeBuiltins@