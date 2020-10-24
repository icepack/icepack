SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = icepack
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf build source/icepack* source/modules.rst

.PHONY: help clean Makefile

html: Makefile
	sphinx-apidoc --force --maxdepth=1 --module-first -o source ../icepack
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
