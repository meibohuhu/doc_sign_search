#!/bin/bash
# SSH Connection Diagnostic Script
# This script helps diagnose SSH connection issues to the cluster

echo "🔍 SSH Connection Diagnostic Tool"
echo "================================="
echo ""

# Check if we're already on the cluster
echo "1. Checking current hostname..."
hostname
echo ""

# Check SSH configuration
echo "2. Checking SSH configuration..."
if [ -f ~/.ssh/config ]; then
    echo "SSH config file exists:"
    cat ~/.ssh/config
else
    echo "No SSH config file found"
fi
echo ""

# Check SSH keys
echo "3. Checking SSH keys..."
ls -la ~/.ssh/
echo ""

# Test basic connectivity
echo "4. Testing network connectivity..."
if command -v ping >/dev/null 2>&1; then
    echo "Testing ping to common hosts..."
    ping -c 1 8.8.8.8 >/dev/null 2>&1 && echo "✅ Internet connectivity: OK" || echo "❌ Internet connectivity: FAILED"
else
    echo "ping command not available"
fi
echo ""

# Check if we can resolve hostnames
echo "5. Testing DNS resolution..."
if command -v nslookup >/dev/null 2>&1; then
    nslookup google.com >/dev/null 2>&1 && echo "✅ DNS resolution: OK" || echo "❌ DNS resolution: FAILED"
else
    echo "nslookup command not available"
fi
echo ""

# Check SSH agent
echo "6. Checking SSH agent..."
if [ -n "$SSH_AUTH_SOCK" ]; then
    echo "SSH agent is running"
    ssh-add -l 2>/dev/null && echo "✅ SSH keys loaded" || echo "⚠️  No SSH keys loaded"
else
    echo "SSH agent is not running"
fi
echo ""

# Check for common SSH issues
echo "7. Common SSH troubleshooting tips:"
echo "   - Make sure your SSH key is added to the cluster's authorized_keys"
echo "   - Check if the cluster requires VPN connection"
echo "   - Verify the cluster hostname/IP address"
echo "   - Check if there are firewall restrictions"
echo "   - Ensure your SSH key has the correct permissions (600)"
echo ""

echo "8. To test SSH connection manually, try:"
echo "   ssh -v username@cluster-hostname"
echo "   (Replace username and cluster-hostname with actual values)"
echo ""

echo "🔧 If SSH is still not working:"
echo "   1. Contact your cluster administrator"
echo "   2. Check cluster status page"
echo "   3. Verify your account is active"
echo "   4. Try connecting from a different network"
echo ""
