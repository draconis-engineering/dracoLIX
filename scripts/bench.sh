#!/bin/bash

# --------------------------------------- #

echo "Building testing executables..."

cd "C:/Prosjekter/dracolix/testing/executables" || exit 1

make

echo "Done"

# --------------------------------------- #

echo "Benching against NumPy..."
cd "C:/Prosjekter/dracolix/bin" || exit 1

for file in ./*; do
    # Check if it is a regular file and executable
    if [ -f "\(file" ] && [ -x "\)file" ]; then
        echo "Found executable: \$file"

        # Ensure the output directory exists
        mkdir -p ../temp

        # Execute the binary and redirect output to a text file
        "file" > "../temp/{file##*/}.txt" 2>&1

        echo "Success"
    fi
done
echo "Done"

# --------------------------------------- #

echo "Documenting test results..."
cd "C:/Prosjekter/dracolix/results" || exit 1

for file in ./*; do
    # Check if it is a regular file
    if [ -f "\$file" ]; then
        echo "Found text file: \$file"

        # Ensure the destination directory exists
        mkdir -p ../temp

        # Convert text file to markdown by wrapping content in code blocks
        echo "### Test Results for \({file##*/}" > "../temp/\){file##*/}.md"
        echo '```text' >> "../temp/\${file##*/}.md"
        cat "file" >> "../temp/{file##*/}.md"
        echo '```' >> "../temp/${file##*/}.md"
    fi
done
echo "Done"
