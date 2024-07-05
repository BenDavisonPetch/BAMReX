#!/bin/bash
for d in output/*/*/*/ ; do (cd ${d} ; ls -1 plt?????/Header | tee headers.visit); done