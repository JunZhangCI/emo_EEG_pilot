#!/bin/bash
git config filter.strip-notebook-output.clean \
  "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
  
git config filter.strip-notebook-output.smudge cat

echo "Git filter for stripping notebook output installed."