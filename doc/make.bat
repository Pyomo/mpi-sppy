@ECHO OFF

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
if "%BUILDDIR%" == "" (
	set BUILDDIR=build
)
if "%SOURCEDIR%" == "" (
	set SOURCEDIR=src
)

if "%1" == "" goto help

if "%1" == "help" (
	:help
    %SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%

	goto end
) else (
	%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

	goto end
)

:end
