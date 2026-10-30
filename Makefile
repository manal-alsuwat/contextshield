# Unix make; on Windows use:  python make.py <target>
PY ?= .venv/bin/python
.PHONY: all dataset bench core optional summary trace leak test dry demo
all:      ; $(PY) make.py all
dataset:  ; $(PY) make.py dataset
bench:    ; $(PY) make.py bench
core:     ; $(PY) make.py core
optional: ; $(PY) make.py optional
summary:  ; $(PY) make.py summary
trace:    ; $(PY) make.py trace
leak:     ; $(PY) make.py leak
test:     ; $(PY) make.py test
dry:      ; $(PY) make.py dry
demo:     ; $(PY) -m streamlit run demo/app.py
