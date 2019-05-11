time xargs -P 4 -I {} sh -c 'eval "$1"' - {} <<'EOF'
./main.py 4 1
./main.py 4 2
./main.py 4 3
EOF
