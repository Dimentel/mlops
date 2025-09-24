#!/bin/bash

# open-jupyter.sh
echo "=== Jupyter Lab Auto-Opener ==="

# Simple Docker check
if ! docker version > /dev/null 2>&1; then
    echo "❌ Please start Docker first!"
    exit 1
fi
echo "✅ Docker: OK"

# Try to get token
echo "🔑 Getting access token..."
token=""

# Method 1: access file
if access_content=$(docker-compose exec jupyter cat /home/jovyan/ACCESS.md 2>/dev/null); then
    token_line=$(echo "$access_content" | grep "Token")
    if [ -n "$token_line" ]; then
        # Делим строку по двоеточию и берём вторую часть
        token=$(echo "$token_line" | cut -d':' -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        echo "✅ Token found in access file: $token"
    fi
fi

# Method 2: From logs
if [ -z "$token" ]; then
    token_line=$(docker-compose logs jupyter 2>/dev/null | grep "token=" | head -1)
    if echo "$token_line" | grep -q "token="; then
        token=$(echo "$token_line" | grep -o "token=[a-f0-9]*" | cut -d'=' -f2)
        echo "✅ Token found in logs: $token"
    fi
fi

# Construct URL
if [ -n "$token" ]; then
    url="http://localhost:8855/lab?token=$token"
    echo "✅ Token found, URL constructed"
    echo "🌐 $url"
else
    url="http://localhost:8855/lab"
    echo "⚠️ No token found, opening without authentication"
fi

echo "🚀 Opening: $url"
xdg-open "$url" || open "$url" || start "$url" 2>/dev/null

echo ""
echo "💡 If page doesn't load, check: docker-compose ps"
echo ""
echo "🔧 Useful commands:"
echo "  Check logs: docker-compose logs jupyter"
echo "  Check status: docker-compose ps"
echo "  Restart: docker-compose restart jupyter"