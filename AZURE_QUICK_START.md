# Azure DevOps Integration - Quick Start Guide

## What's New?

Your TestDriver MCP Chat now has **comprehensive Azure DevOps integration** with support for 5 different authentication methods and encrypted credential storage.

## Getting Started (2 minutes)

### Step 1: Open Azure Integration
Click the **☁ Azure Integration** button in the chat header (next to Model Config)

### Step 2: Select Authentication Method
Choose one of these 5 options:

| Method | Use When | Complexity |
|--------|----------|-----------|
| **Personal Access Token (PAT)** | Individual development, quick setup | ⭐ Easy |
| **Managed Identity** | Running on Azure infrastructure (VM, App Service, AKS) | ⭐⭐ Medium |
| **SSH Public Key** | CI/CD pipelines, high security requirements | ⭐⭐ Medium |
| **Service Principal** | Enterprise automation at scale, multi-tenant | ⭐⭐⭐ Complex |
| **OAuth 2.0 (OIDC)** | Browser-based interactive authentication | ⭐⭐⭐ Complex |

### Step 3: Enter Your Credentials
Follow the on-screen guidance for your selected method. Each field has helpful documentation.

### Step 4: Test Connection
Click "Test Connection" to validate your settings before saving.

### Step 5: Save Configuration
Click "Save Configuration" to encrypt and store your credentials locally.

## Authentication Methods Explained

### 1️⃣ Personal Access Token (PAT) - RECOMMENDED FOR BEGINNERS

**What You Need:**
- Azure DevOps Organization URL (e.g., `https://dev.azure.com/myorg`)
- Personal Access Token from Azure DevOps

**How to Create a PAT:**
1. Go to: https://dev.azure.com/{organization}/_usersSettings/tokens
2. Click "New Token"
3. Name: "TestDriver MCP"
4. Scopes: Check "Code" → "Read & Write"
5. Expiration: Choose based on your policy (90 days recommended)
6. Copy the token (you won't see it again!)

**Best For:**
- Individual developers
- Quick setup and testing
- Development environments

**Security:**
- Single user token
- Can be regenerated anytime
- Store in vault for production

---

### 2️⃣ Managed Identity - RECOMMENDED FOR AZURE DEPLOYMENTS

**What You Need:**
- Running on Azure infrastructure (VM, App Service, AKS, Function, Container)
- System-assigned or User-assigned Managed Identity

**Setup (if using Azure VM):**
1. Ensure VM has System-assigned Managed Identity enabled
2. Enter your organization URL
3. Select "System-assigned" identity type
4. TestDriver will automatically use the VM's identity

**Best For:**
- Running on Azure infrastructure
- No credential management needed
- Automatic token rotation
- Enterprise security

**Security:**
- Credentials never touch disk
- Automatic renewal
- Audit-logged in Azure

---

### 3️⃣ SSH Public Key - RECOMMENDED FOR CI/CD

**What You Need:**
- SSH public and private key pair
- Optional: passphrase protecting private key

**How to Create SSH Keys:**
```bash
# On your machine or CI/CD runner:
ssh-keygen -t rsa -b 4096 -f testdriver_key -N ""

# Two files are created:
# testdriver_key (private key - keep secret!)
# testdriver_key.pub (public key - upload to Azure)
```

**Setup:**
1. Paste public key in "Public Key" field
2. Paste private key in "Private Key" field
3. Add passphrase if key has one
4. Enter organization URL
5. TestDriver will use SSH authentication

**Best For:**
- CI/CD pipelines
- Git operations
- High security requirements
- Team environments

**Security:**
- Key-based authentication (not password)
- Passphrase protection optional but recommended
- Can be rotated easily
- Industry standard

---

### 4️⃣ Service Principal - RECOMMENDED FOR ENTERPRISE

**What You Need:**
- Tenant ID (your Azure AD directory)
- Client ID (App Registration ID)
- Client Secret (password for the app)
- Service Principal must have DevOps permissions

**Setup (5 minutes):**
1. Go to: https://portal.azure.com → Azure Active Directory → App Registrations
2. Click "New registration"
3. Name: "TestDriver MCP"
4. Supported account types: "Accounts in this organizational directory only"
5. Click Register
6. Copy "Application (client) ID" → paste as Client ID
7. Go to Certificates & secrets → New client secret
8. Copy value → paste as Client Secret
9. Grant permissions to Azure DevOps

**Best For:**
- Enterprise automation
- Multi-tenant deployments
- Compliance requirements
- Team deployments

**Security:**
- Service account (not personal)
- Audit trail in Azure
- Can enforce MFA
- Secrets can be rotated

---

### 5️⃣ OAuth 2.0 (OIDC) - RECOMMENDED FOR INTERACTIVE

**What You Need:**
- OAuth app registration in Azure AD
- Client ID and Client Secret
- Redirect URI
- User MFA (optional)

**Setup (10 minutes):**
1. Register OAuth app in Azure AD (see Service Principal setup)
2. Set Redirect URI to your TestDriver instance
3. Request user login via OAuth
4. TestDriver receives access token
5. Use token for Azure DevOps operations

**Best For:**
- Web applications
- User-interactive scenarios
- Single sign-on (SSO)
- When you want user-initiated auth

**Security:**
- User-specific permissions
- Modern OAuth 2.0 standard
- Support for MFA
- Token expiration policies

---

## Common Questions

### Q: Which method should I use?
**A:** 
- **Just testing?** → Use **Personal Access Token (PAT)**
- **On Azure infrastructure?** → Use **Managed Identity**
- **CI/CD pipeline?** → Use **SSH Public Key**
- **Large enterprise?** → Use **Service Principal**
- **User login required?** → Use **OAuth 2.0**

### Q: Where are my credentials stored?
**A:** Encrypted on your local disk in `.azure_config` file with encryption key in `.azure_key`. No credentials in code, logs, or memory.

### Q: Can I change authentication methods?
**A:** Yes! Just open Azure Integration again and select a different method. The new one will replace the previous one.

### Q: What if my credentials expire?
**A:** 
- **PAT**: Generate new PAT in Azure DevOps, update in TestDriver
- **SSH**: Generate new key pair, update in TestDriver
- **Service Principal**: Generate new secret, update in TestDriver
- **Managed Identity**: Expires automatically, renewed by Azure
- **OAuth**: Refresh token automatically

### Q: Is this secure?
**A:** Yes! 
- All credentials encrypted with AES-128 Fernet cipher
- File permissions restricted to owner only
- No credentials logged to console
- No transmission in clear text (HTTPS recommended for production)

### Q: Can I use this on production?
**A:** Yes, with these recommendations:
- Use **Managed Identity** if on Azure
- Use **Service Principal** for enterprise automation
- Use HTTPS (not HTTP) for all connections
- Regularly rotate credentials
- Consider moving credentials to Azure Key Vault

---

## Troubleshooting

### Connection Test Fails
1. Check organization URL is correct
2. For PAT: verify token is not expired
3. Verify network connectivity to Azure
4. Check exact error message for details

### "Configuration saved successfully" but credentials not working
1. Verify Test Connection passed before saving
2. Credentials are encrypted locally, but Azure DevOps validation happens during test
3. Try Test Connection again to verify

### Button disappeared
1. Reload the chat page (Ctrl+R)
2. Check browser console (F12) for errors
3. Verify server is running (refresh chat)

---

## Next Steps

1. **Choose your authentication method** above
2. **Click Azure Integration button** in chat
3. **Follow the 3-step wizard**
4. **Test and save your configuration**
5. **Start using Azure integration features!**

---

## Additional Resources

- **Azure DevOps Documentation**: https://docs.microsoft.com/azure/devops
- **Azure AD Application**: https://portal.azure.com
- **Personal Access Tokens**: https://dev.azure.com/{org}/_usersSettings/tokens
- **SSH Key Setup**: https://docs.github.com/authentication/connecting-to-github-with-ssh

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed implementation guide in `AZURE_INTEGRATION_COMPLETE.md`
3. Check server logs for error details
4. Verify Azure DevOps connectivity separately

---

**Happy Testing! 🚀**
