time xargs -P 16 -I {} sh -c 'eval "$1"' - {} <<'EOF'
./main.py 17 1
./main.py 17 2
./main.py 17 3
./main.py 17 4
./main.py 17 5
./main.py 17 6
./main.py 17 7
./main.py 17 8
./main.py 17 9
./main.py 17 10
./main.py 17 11
./main.py 17 12
./main.py 17 13
./main.py 17 14
./main.py 17 15
./main.py 17 16
EOF
