# ============================================================================
# PET Library Configuration
# Copyright (c) 2025 AutoPoly Contributors
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# PET (Polyhedral Extraction Tool) is a library for extracting polyhedral models from C source code.
# It uses Clang for parsing C code and provides a polyhedral representation that can be used by PPCG.

message(STATUS "PET source dir: ${PET_DIR}")

# Create gitversion.h file (required by PET)
execute_process(
    COMMAND git describe --always 
    WORKING_DIRECTORY ${PET_DIR}
    OUTPUT_VARIABLE PET_GIT_HEAD_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT PET_GIT_HEAD_VERSION)
    set(PET_GIT_HEAD_VERSION "unknown")
endif()
file(WRITE ${PET_DIR}/gitversion.h "#define GIT_HEAD_ID \"${PET_GIT_HEAD_VERSION}\"")

# Set PET version
string(REGEX REPLACE "^pet-([0-9.]+).*" "\\1" PET_VERSION "${PET_GIT_HEAD_VERSION}")
if(NOT PET_VERSION OR PET_VERSION STREQUAL PET_GIT_HEAD_VERSION)
    set(PET_VERSION "0.11.8")  # fallback version
endif()
message(STATUS "PET version: ${PET_VERSION}")

########################################################################
# Check compiler characteristics to set PET compilation options        #
########################################################################
message(STATUS "Checking Clang features for PET compatibility")

# Set Clang prefix
if(NOT "${PET_CLANG_PREFIX}" STREQUAL "")
    set(CLANG_PREFIX "${PET_CLANG_PREFIX}")
else()
    # Try to find clang executable and derive prefix
    find_program(CLANG_EXECUTABLE clang)
    if(CLANG_EXECUTABLE)
        get_filename_component(CLANG_BIN_DIR ${CLANG_EXECUTABLE} DIRECTORY)
        get_filename_component(CLANG_PREFIX ${CLANG_BIN_DIR} DIRECTORY)
    else()
        set(CLANG_PREFIX "/usr")
    endif()
endif()
message(STATUS "CLANG_PREFIX: ${CLANG_PREFIX}")

# Check for standard headers
check_include_file("dlfcn.h" HAVE_DLFCN_H)
check_include_file("inttypes.h" HAVE_INTTYPES_H)
check_include_file("stdint.h" HAVE_STDINT_H)
check_include_file("stdio.h" HAVE_STDIO_H)
check_include_file("stdlib.h" HAVE_STDLIB_H)
check_include_file("string.h" HAVE_STRING_H)
check_include_file("strings.h" HAVE_STRINGS_H)
check_include_file("sys/stat.h" HAVE_SYS_STAT_H)
check_include_file("sys/types.h" HAVE_SYS_TYPES_H)
check_include_file("unistd.h" HAVE_UNISTD_H)

# Set STDC_HEADERS if all standard headers are available
if(HAVE_STDIO_H AND HAVE_STDLIB_H AND HAVE_STRING_H AND HAVE_STDINT_H)
    set(STDC_HEADERS 1)
endif()

# Check for Clang-specific headers
check_cxx_source_compiles("
    #include <clang/Basic/SourceLocation.h>
    int main() { return 0; }" 
    HAVE_BASIC_SOURCELOCATION_H)

check_cxx_source_compiles("
    #include <clang/Basic/DiagnosticOptions.h>
    int main() { return 0; }" 
    HAVE_BASIC_DIAGNOSTICOPTIONS_H)

check_cxx_source_compiles("
    #include <clang/Lex/HeaderSearchOptions.h>
    int main() { return 0; }" 
    HAVE_LEX_HEADERSEARCHOPTIONS_H)

check_cxx_source_compiles("
    #include <clang/Lex/PreprocessorOptions.h>
    int main() { return 0; }" 
    HAVE_LEX_PREPROCESSOROPTIONS_H)

check_cxx_source_compiles("
    #include <clang/Basic/LangStandard.h>
    int main() { return 0; }" 
    HAVE_CLANG_BASIC_LANGSTANDARD_H)

check_cxx_source_compiles("
    #include <llvm/Option/Arg.h>
    int main() { return 0; }" 
    HAVE_LLVM_OPTION_ARG_H)

check_cxx_source_compiles("
    #include <llvm/ADT/OwningPtr.h>
    int main() { return 0; }" 
    HAVE_ADT_OWNINGPTR_H)

check_cxx_source_compiles("
    #include <llvm/TargetParser/Host.h>
    int main() { return 0; }" 
    HAVE_TARGETPARSER_HOST_H)

# Check for method name variations (backwards compatibility)
check_cxx_source_compiles("
    #include <llvm/TargetParser/Host.h>
    #include <clang/Basic/TargetInfo.h>
    int main() {
        auto T = llvm::sys::getDefaultTargetTriple();
        return 0;
    }"
    HAVE_GETDEFAULTTARGETTRIPPLE)
if(NOT HAVE_GETDEFAULTTARGETTRIPPLE)
    set(getDefaultTargetTriple "getHostTriple")
endif()

# Check for DiagnosticInfo vs Diagnostic compatibility
check_cxx_source_compiles("
    #include <clang/Basic/Diagnostic.h>
    int main() {
        clang::DiagnosticInfo* DI = nullptr;
        return 0;
    }"
    HAVE_DIAGNOSTIC_INFO)

if(NOT HAVE_DIAGNOSTIC_INFO)
    set(DiagnosticInfo "Diagnostic")
endif()

# Check for DiagnosticsEngine vs Diagnostic
check_cxx_source_compiles("
    #include <clang/Basic/Diagnostic.h>
    int main() {
        clang::DiagnosticsEngine* DE = nullptr;
        return 0;
    }"
    HAVE_DIAGNOSTICS_ENGINE)

if(NOT HAVE_DIAGNOSTICS_ENGINE)
    set(DiagnosticsEngine "Diagnostic")
endif()

# Check HandleTopLevelDecl return type
check_cxx_source_compiles("
    #include <clang/AST/ASTConsumer.h>
    #include <clang/AST/DeclGroup.h>
    class TestConsumer : public clang::ASTConsumer {
    public:
        bool HandleTopLevelDecl(clang::DeclGroupRef D) override { return true; }
    };
    int main() { return 0; }"
    HANDLETOPLEVELRETURN_IS_BOOL)

if(HANDLETOPLEVELRETURN_IS_BOOL)
    set(HandleTopLevelDeclReturn "bool")
    set(HandleTopLevelDeclContinue "true")
else()
    set(HandleTopLevelDeclReturn "void")
    set(HandleTopLevelDeclContinue "")
endif()

# Check if CreateTargetInfo takes shared_ptr
check_cxx_source_compiles("
    #include <clang/Basic/TargetInfo.h>
    #include <clang/Basic/Diagnostic.h>
    #include <memory>
    int main() {
        std::shared_ptr<clang::TargetOptions> TO;
        clang::DiagnosticsEngine DE(nullptr, nullptr);
        clang::TargetInfo::CreateTargetInfo(DE, TO);
        return 0;
    }"
    CREATETARGETINFO_TAKES_SHARED_PTR)

# Check if CreateTargetInfo takes pointer
if(NOT CREATETARGETINFO_TAKES_SHARED_PTR)
    check_cxx_source_compiles("
        #include <clang/Basic/TargetInfo.h>
        #include <clang/Basic/Diagnostic.h>
        int main() {
            clang::TargetOptions TO;
            clang::DiagnosticsEngine DE(nullptr, nullptr);
            clang::TargetInfo::CreateTargetInfo(DE, &TO);
            return 0;
        }"
        CREATETARGETINFO_TAKES_POINTER)
endif()

# Check if Driver::BuildCompilation takes ArrayRef
check_cxx_source_compiles("
    #include <clang/Driver/Driver.h>
    #include <llvm/ADT/ArrayRef.h>
    #include <clang/Basic/DiagnosticIDs.h>
    #include <clang/Basic/DiagnosticOptions.h>
    #include <llvm/Support/VirtualFileSystem.h>
    
    int main() { 
        clang::DiagnosticIDs diagIDs;
        clang::DiagnosticOptions diagOpts;
        clang::DiagnosticsEngine diagEngine(&diagIDs, &diagOpts);
        clang::driver::Driver D(\"\", \"\", diagEngine);
        llvm::ArrayRef<const char*> Args;
        D.BuildCompilation(Args);
        return 0; 
    }"
    USE_ARRAYREF)

# Check if CompilerInvocation::CreateFromArgs takes ArrayRef
check_cxx_source_compiles("
    #include <clang/Frontend/CompilerInvocation.h>
    #include <llvm/ADT/ArrayRef.h>
    #include <memory>

    int main() { 
        llvm::ArrayRef<const char*> Args;
        clang::DiagnosticsEngine DE(nullptr, nullptr);
        auto invocation = std::make_shared<clang::CompilerInvocation>();
        clang::CompilerInvocation::CreateFromArgs(*invocation, Args, DE);
        return 0; 
    }"
    CREATE_FROM_ARGS_TAKES_ARRAYREF)

check_cxx_source_compiles("
    #include <clang/Lex/HeaderSearchOptions.h>
    int main() {
        using namespace clang;
        clang::HeaderSearchOptions HSO;
        HSO.AddPath(\"\", clang::frontend::Angled, false, false);
        return 0;
    }" ADDPATH_TAKES_4_ARGUMENTS)

# Check if CompilerInstance::createPreprocessor takes TranslationUnitKind
check_cxx_source_compiles("
    #include <clang/Frontend/CompilerInstance.h>
    int main() {
        clang::CompilerInstance CI;
        CI.createPreprocessor(clang::TU_Complete);
        return 0;
    }"
    CREATEPREPROCESSOR_TAKES_TUKIND)

# Check for DecayedType availability
check_cxx_source_compiles("
    #include <clang/AST/Type.h>
    int main() { 
        clang::DecayedType* DT = nullptr;
        return 0; 
    }"
    HAVE_DECAYEDTYPE)

# Check if SourceManager has setMainFileID method
check_cxx_source_compiles("
    #include <clang/Basic/SourceManager.h>
    int main() {
        clang::SourceManager* SM = nullptr;
        SM->setMainFileID(clang::FileID());
        return 0;
    }"
    HAVE_SETMAINFILEID)

# Check if SourceManager has findLocationAfterToken method
check_cxx_source_compiles("
    #include <clang/Lex/Lexer.h>
    #include <clang/Basic/SourceManager.h>
    
    int main() {
        clang::SourceManager* SM = nullptr;
        clang::SourceLocation SL;
        clang::Lexer::findLocationAfterToken(SL, clang::tok::semi, *SM, clang::LangOptions(), false);
        return 0;
    }"
    HAVE_FINDLOCATIONAFTERTOKEN)

# Check if SourceManager has translateLineCol method
check_cxx_source_compiles("
    #include <clang/Basic/SourceManager.h>
    int main() {
        clang::SourceManager* SM = nullptr;
        SM->translateLineCol(clang::FileID(), 1, 1);
        return 0;
    }"
    HAVE_TRANSLATELINECOL)

# Check if ASTContext::getTypeInfo returns TypeInfo
check_cxx_source_compiles("
    #include <clang/AST/ASTContext.h>
    int main() {
        clang::ASTContext* AC = nullptr;
        clang::TypeInfo TI = AC->getTypeInfo(clang::QualType());
        return 0;
    }"
    GETTYPEINFORETURNSTYPEINFO)

# Check for setLangDefaults location and signature
check_cxx_source_compiles("
    #include <clang/Basic/LangOptions.h>
    #include <clang/Basic/TargetInfo.h>
    #include <vector>
    #include <string>

    int main() { 
        clang::LangOptions LO;
        std::vector<std::string> Includes;
        clang::LangOptions::setLangDefaults(LO, clang::Language::C, llvm::Triple(), 
                                        Includes, clang::LangStandard::lang_unspecified); 
        return 0; 
    }" 
    SETLANGDEFAULTS_IS_LANGOPTIONS)

if(SETLANGDEFAULTS_IS_LANGOPTIONS)
    set(SETLANGDEFAULTS "LangOptions")
else()
    set(SETLANGDEFAULTS "CompilerInvocation")
endif()

# Check for InputKind::Language definition
check_cxx_source_compiles("
    #include <clang/Frontend/FrontendOptions.h>
    int main() { 
        clang::Language L = clang::Language::C; 
        return 0; 
    }" 
    IK_C_IS_LANGUAGE_C)
if(IK_C_IS_LANGUAGE_C)
    set(IK_C "Language::C")
else()
    set(IK_C "InputKind::C")
endif()

# Check if setLangDefaults takes 5 arguments
check_cxx_source_compiles("
    #include <clang/Basic/TargetOptions.h>
    #include <clang/Lex/PreprocessorOptions.h>
    #include <clang/Frontend/CompilerInstance.h>
    #include <string>
    #include <vector>

    struct setLangDefaultsArg4 {
        setLangDefaultsArg4(clang::PreprocessorOptions &PO) : PO(PO) {}
        operator clang::PreprocessorOptions &() { return PO; }
        operator std::vector<std::string> &() { return PO.Includes; }

        clang::PreprocessorOptions &PO;
    };
    int main() {
        using namespace clang;
        CompilerInstance *Clang;
        TargetOptions TO;
        llvm::Triple T(TO.Triple);
        PreprocessorOptions PO;
        ${SETLANGDEFAULTS}::setLangDefaults(Clang->getLangOpts(), ${IK_C},
                T, setLangDefaultsArg4(PO),
                LangStandard::lang_unspecified);
        return 0;
    }"
    SETLANGDEFAULTS_TAKES_5_ARGUMENTS)

# Check if CompilerInstance::setInvocation takes shared_ptr
check_cxx_source_compiles("
    #include <clang/Frontend/CompilerInstance.h>
    #include <clang/Frontend/CompilerInvocation.h>
    #include <memory>
    int main() {
        clang::CompilerInstance CI;
        auto invocation = std::make_shared<clang::CompilerInvocation>();
        CI.setInvocation(invocation);
        return 0;
    }"
    SETINVOCATION_TAKES_SHARED_PTR)

# Check for getBeginLoc/getEndLoc methods
check_cxx_source_compiles("
    #include <clang/AST/Decl.h>
    int main() {
        clang::FunctionDecl* FD = nullptr;
        clang::SourceLocation begin = FD->getBeginLoc();
        clang::SourceLocation end = FD->getEndLoc();
        return 0;
    }"
    HAVE_BEGIN_END_LOC)

# Check if DiagnosticsEngine has setDiagnosticGroupWarningAsError
check_cxx_source_compiles("
    #include <clang/Basic/Diagnostic.h>
    int main() {
        clang::DiagnosticsEngine DE(nullptr, nullptr);
        DE.setDiagnosticGroupWarningAsError(\"test\", true);
        return 0;
    }"
    HAVE_SET_DIAGNOSTIC_GROUP_WARNING_AS_ERROR)

# Check for ext_implicit_function_decl_c99
check_cxx_source_compiles("
    #include <clang/Basic/DiagnosticCategories.h>
    int main() {
        auto DiagID = clang::diag::ext_implicit_function_decl_c99;
        return 0;
    }"
    HAVE_EXT_IMPLICIT_FUNCTION_DECL_C99)
if(NOT HAVE_EXT_IMPLICIT_FUNCTION_DECL_C99)
    set(ext_implicit_function_decl_c99 "ext_implicit_function_decl")
endif()

# Check for StmtRange class
check_cxx_source_compiles("
    #include <clang/AST/StmtIterator.h>
    int main() { 
        clang::StmtRange SR;
        return 0; 
    }"
    HAVE_STMTRANGE)

# Check for nested ArraySizeModifier in ArrayType
check_cxx_source_compiles("
    #include <clang/AST/Type.h>
    int main() {
        clang::ArrayType::ArraySizeModifier ASM;
        return 0;
    }"
    USE_NESTED_ARRAY_SIZE_MODIFIER)

# Check if CompilerInstance::createDiagnostics takes argc/argv
check_cxx_source_compiles("
    #include <clang/Frontend/CompilerInstance.h>
    int main() {
        using namespace clang;
        CompilerInstance *Clang;
        Clang->createDiagnostics();
        return 0;
    }"
    NO_CREATEDIAGNOSTICS_TAKES_ARG)
if(NOT NO_CREATEDIAGNOSTICS_TAKES_ARG)
    set(CREATEDIAGNOSTICS_TAKES_ARG "1")
endif()

# Check for DiagnosticConsumer vs DiagnosticClient
check_cxx_source_compiles("
    #include <clang/Basic/Diagnostic.h>
    int main() {
        clang::DiagnosticConsumer* DC = nullptr;
        return 0;
    }"
    HAVE_DIAGNOSTIC_CONSUMER)
if(NOT HAVE_DIAGNOSTIC_CONSUMER)
    set(DiagnosticConsumer "DiagnosticClient")
endif()

# Check for TypedefNameDecl vs TypedefDecl
check_cxx_source_compiles("
    #include <clang/AST/Decl.h>
    int main() {
        clang::TypedefNameDecl* TND = nullptr;
        return 0;
    }"
    HAVE_TYPEDEFNAMEDECL)
if(NOT HAVE_TYPEDEFNAMEDECL)
    set(TypedefNameDecl "TypedefDecl")
endif()

# Check for PragmaIntroducer vs PragmaIntroducerKind
check_cxx_source_compiles("
    #include <clang/Lex/Pragma.h>
    int main() {
        clang::PragmaIntroducer PI;
        return 0;
    }"
    HAVE_PRAGMAINTRODUCER)
if(NOT HAVE_PRAGMAINTRODUCER)
    set(PragmaIntroducer "PragmaIntroducerKind")
endif()

# Check for expansion vs instantiation methods
check_cxx_source_compiles("
    #include <clang/Basic/SourceManager.h>
    int main() {
        using namespace clang;
        SourceManager *sm = nullptr;
        SourceLocation sl;
        sm->getExpansionColumnNumber(sl);
        return 0;
    }"
    HAVE_GETEXPANSIONCOLUMNNUMBER)

check_cxx_source_compiles("
    #include <clang/Basic/SourceManager.h>
    int main() {
        using namespace clang;
        SourceManager *sm = nullptr;
        SourceLocation sl;
        sm->getExpansionLineNumber(sl);
        return 0;
    }"
    HAVE_GETEXPANSIONLINENUMBER)

check_cxx_source_compiles("
    #include <clang/Basic/SourceManager.h>
    int main() {
        using namespace clang;
        SourceManager *sm = nullptr;
        SourceLocation sl;
        sm->getExpansionLoc(sl);
        return 0;
    }"
    HAVE_GETEXPANSIONLOC)
if(NOT HAVE_GETEXPANSIONCOLUMNNUMBER)
    set(getExpansionColumnNumber "getInstantiationColumnNumber")
endif()
if(NOT HAVE_GETEXPANSIONLINENUMBER)
    set(getExpansionLineNumber "getInstantiationLineNumber")
endif()
if(NOT HAVE_GETEXPANSIONLOC)
    set(getExpansionLoc "getInstantiationLoc")
endif()

# Check for getLangOpts vs getLangOptions
check_cxx_source_compiles("
    #include <clang/Frontend/CompilerInstance.h>
    int main() {
        clang::CompilerInstance CI;
        const clang::LangOptions& LO = CI.getLangOpts();
        return 0;
    }"
    HAVE_GETLANGOPTS)
if(NOT HAVE_GETLANGOPTS)
    set(getLangOpts "getLangOptions")
endif()

# Check for getLocWithOffset vs getFileLocWithOffset
check_cxx_source_compiles("
    #include <clang/Basic/SourceManager.h>
    int main() {
        clang::SourceLocation *loc = nullptr;
        auto res = loc->getLocWithOffset(1);
        return 0;
    }"
    HAVE_GETLOCWITHOFFSET)
if(NOT HAVE_GETLOCWITHOFFSET)
    set(getLocWithOffset "getFileLocWithOffset")
endif()

# Check for getReturnType vs getResultType
check_cxx_source_compiles("
    #include <clang/AST/Type.h>
    int main() {
        clang::FunctionType* FT = nullptr;
        clang::QualType QT = FT->getReturnType();
        return 0;
    }"
    HAVE_GETRETURNTYPE)
if(NOT HAVE_GETRETURNTYPE)
    set(getReturnType "getResultType")
endif()

# Check for getTypedefNameForAnonDecl vs getTypedefForAnonDecl
check_cxx_source_compiles("
    #include <clang/AST/Decl.h>
    int main() {
        clang::RecordDecl *decl = nullptr;
        auto res = decl->getTypedefNameForAnonDecl();
        return 0;
    }"
    HAVE_GETTYPEDEFNAMEFORANONDECL)
if(NOT HAVE_GETTYPEDEFNAMEFORANONDECL)
    set(getTypedefNameForAnonDecl "getTypedefForAnonDecl")
endif()

# Check for initializeBuiltins vs InitializeBuiltins
check_cxx_source_compiles("
    #include <clang/Basic/Builtins.h>
    #include <clang/Basic/LangOptions.h>
    #include <clang/Lex/Preprocessor.h>
    int main() {
        clang::LangOptions LO;
        clang::Preprocessor* PP = nullptr;
        PP->getBuiltinInfo().initializeBuiltins(PP->getIdentifierTable(), LO);
        return 0;
    }"
    HAVE_INITIALIZEBUILTINS)
if(NOT HAVE_INITIALIZEBUILTINS)
    set(initializeBuiltins "InitializeBuiltins")
endif()

# Handle true/false for boolean flags in CMake
if(STDC_HEADERS)
    set(STDC_HEADERS 1)
endif()
if(HAVE_DLFCN_H)
    set(HAVE_DLFCN_H 1)
endif()
if(HAVE_INTTYPES_H)
    set(HAVE_INTTYPES_H 1)
endif()
if(HAVE_STDINT_H)
    set(HAVE_STDINT_H 1)
endif()
if(HAVE_STDIO_H)
    set(HAVE_STDIO_H 1)
endif()
if(HAVE_STDLIB_H)
    set(HAVE_STDLIB_H 1)
endif()
if(HAVE_STRINGS_H)
    set(HAVE_STRINGS_H 1)
endif()
if(HAVE_STRING_H)
    set(HAVE_STRING_H 1)
endif()
if(HAVE_SYS_STAT_H)
    set(HAVE_SYS_STAT_H 1)
endif()
if(HAVE_SYS_TYPES_H)
    set(HAVE_SYS_TYPES_H 1)
endif()
if(HAVE_UNISTD_H)
    set(HAVE_UNISTD_H 1)
endif()

# Create PET configuration header file
configure_file("${AUTOPOLY_SOURCE_DIR}/cmake/pet_config.h.cmake" "${AUTOPOLY_BINARY_DIR}/include/pet/config.h")

# Include PET configuration directory
include_directories(pet PRIVATE ${AUTOPOLY_BINARY_DIR}/include/pet)

# Include PET headers directories
include_directories(pet PRIVATE
    ${PET_DIR}/include
    ${ISL_DIR}/include
)

message(STATUS "PET Clang compatibility checks completed")

# Complete PET source file list (according to Makefile.am)
set(PET_SOURCES
    ${PET_DIR}/aff.c
    ${PET_DIR}/array.c
    ${PET_DIR}/clang.cc
    ${PET_DIR}/context.c
    ${PET_DIR}/expr.c
    ${PET_DIR}/expr_arg.c
    ${PET_DIR}/expr_plus.cc
    ${PET_DIR}/filter.c
    ${PET_DIR}/id.cc
    ${PET_DIR}/isl_id_to_pet_expr.c
    ${PET_DIR}/inlined_calls.cc
    ${PET_DIR}/inliner.cc
    ${PET_DIR}/killed_locals.cc
    ${PET_DIR}/loc.c
    ${PET_DIR}/nest.c
    ${PET_DIR}/options.c
    ${PET_DIR}/patch.c
    ${PET_DIR}/pet_expr_to_isl_pw_aff.c
    ${PET_DIR}/print.c
    ${PET_DIR}/tree.c
    ${PET_DIR}/tree2scop.c
    ${PET_DIR}/scan.cc
    ${PET_DIR}/scop.c
    ${PET_DIR}/scop_plus.cc
    ${PET_DIR}/skip.c
    ${PET_DIR}/substituter.cc
    ${PET_DIR}/summary.c
    ${PET_DIR}/value_bounds.c
    ${PET_DIR}/version.cc
    ${PET_DIR}/pet.cc
)

# Build pet library
add_library(pet ${PET_SOURCES})

# Add include directories for PET
if(LLVM_FOUND)
    target_include_directories(pet PRIVATE ${LLVM_INCLUDE_DIRS})
endif()
if(Clang_FOUND)
    target_include_directories(pet PRIVATE ${CLANG_INCLUDE_DIRS})
endif()

# Add required libraries for PET (TODO: fix errors when clang not found)
target_link_libraries(pet isl ${PET_REQUIRED_CLANG_LIBS})
if(GMP_FOUND)
    target_link_libraries(pet ${GMP_LIBRARIES})
endif()
if(MPFR_FOUND)
    target_link_libraries(pet ${MPFR_LIBRARIES})
endif()
if(LLVM_FOUND)
    target_link_libraries(pet ${LLVM_LIBRARIES})
endif()
if(Clang_FOUND)
    target_link_libraries(pet ${CLANG_LIBRARIES})
endif()

# Set compile options to PET
target_compile_options(pet PRIVATE -g
    $<$<COMPILE_LANGUAGE:C>:
        -Wno-sign-compare
        -Wno-cast-qual
        -Wno-discarded-qualifiers
        -Wno-implicit-fallthrough
        -Wno-unused-function
        -Wno-unused-variable
        -Wno-unused-but-set-variable
        -Wno-type-limits
        -Wno-return-type
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wno-sign-compare
        -Wno-cast-qual
        -Wno-implicit-fallthrough
        -Wno-unused-function
        -Wno-unused-variable
        -Wno-unused-but-set-variable
        -Wno-suggest-override
        -Wno-return-type
    >
)

# Set installation rules for PET
install(TARGETS pet
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${PET_DIR}/include/ DESTINATION include/pet)

# Build PET executable
if(BUILD_PET_EXE)
    message(STATUS "Building PET executable from source")
    # Check libyaml library (required by pet executable)
    find_package(YAML REQUIRED)
    if(YAML_FOUND)
        message(STATUS "LibYaml library found")
        message(STATUS "LibYaml include directories: ${YAML_INCLUDE_DIRS}")
        message(STATUS "LibYaml libraries: ${YAML_LIBRARIES}")
    else ()
        message(FATAL_ERROR "LibYaml library not found")
    endif()

    # Build PET executable from source
    add_executable(pet_exe
        ${PET_DIR}/main.c
        ${PET_DIR}/dummy.cc
        ${PET_DIR}/emit.c
    )
    set_target_properties(pet_exe PROPERTIES OUTPUT_NAME "pet")
    target_include_directories(pet_exe PRIVATE
        ${YAML_INCLUDE_DIRS}
    )

    # Link PET executable with required libraries
    target_link_libraries(pet_exe PRIVATE
        pet
        isl
        ${PET_REQUIRED_CLANG_LIBS}
        ${YAML_LIBRARIES}
    )
    if(GMP_FOUND)
        target_link_libraries(pet_exe PRIVATE ${GMP_LIBRARIES})
    endif()
    if(MPFR_FOUND)
        target_link_libraries(pet_exe PRIVATE ${MPFR_LIBRARIES})
    endif()
    if(LLVM_FOUND)
        target_link_libraries(pet_exe PRIVATE ${LLVM_LIBRARIES})
    endif()
    if(Clang_FOUND)
        target_link_libraries(pet_exe PRIVATE ${CLANG_LIBRARIES})
    endif()

    # Set compile options to the binary of PET
    target_compile_options(pet_exe PRIVATE -g
        $<$<COMPILE_LANGUAGE:C>:
            -Wno-sign-compare
            -Wno-cast-qual
            -Wno-discarded-qualifiers
            -Wno-implicit-fallthrough
            -Wno-unused-function
            -Wno-unused-variable
            -Wno-unused-but-set-variable
            -Wno-type-limits
            -Wno-return-type
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wno-sign-compare
            -Wno-cast-qual
            -Wno-implicit-fallthrough
            -Wno-unused-function
            -Wno-unused-variable
            -Wno-unused-but-set-variable
            -Wno-suggest-override
            -Wno-return-type
        >
    )

    # Set installation rules for PET_EXECUTABLE
    install(TARGETS pet_exe
        RUNTIME DESTINATION bin
    )
endif()

# Set PET variables
set(PET_INCLUDE_DIRS ${PET_DIR}/include ${ISL_INCLUDE_DIRS})
set(PET_LIBRARIES pet isl ${PET_REQUIRED_CLANG_LIBS})
