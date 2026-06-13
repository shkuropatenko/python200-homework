# Warmup 08

## Cloud Concepts Question 1

The core economic model of cloud computing is pay-as-you-go pricing. Instead of buying and maintaining physical servers, companies rent computing resources only when they need them.

---

## Cloud Concepts Question 2

Vertical scaling means upgrading one machine by adding more CPU, RAM, or GPU power. Horizontal scaling means adding more machines to distribute the workload.

A company might use horizontal scaling after a viral product launch because many servers can handle traffic together.

A data scientist training a slow ML model might use vertical scaling by choosing a machine with a stronger GPU and more RAM.

A large data pipeline processing thousands of files would use horizontal scaling because the work can be split across multiple machines.

---

## Cloud Concepts Question 3

### Classifications

- Gmail → SaaS because users simply use the software without managing infrastructure.
- Azure Virtual Machines → IaaS because users manage the operating system and software themselves.
- Azure App Service → PaaS because Azure manages most infrastructure while developers deploy applications.
- AWS S3 → PaaS because AWS manages the storage infrastructure.
- GitHub Codespaces → PaaS because the environment is managed for developers.
- Snowflake → SaaS because it provides a managed analytics platform without infrastructure management.

### Definitions

IaaS provides raw infrastructure like virtual machines, storage, and networking. Example: Azure Virtual Machines. The developer manages the operating system, software, and applications.

PaaS provides managed platforms for building and deploying applications. Example: Azure App Service. The developer mainly manages the application code and data.

SaaS provides fully managed software accessible through the internet. Example: Gmail. The provider manages almost everything.

---

## Cloud Concepts Question 4

Managed data platforms like Snowflake or Databricks simplify working with data by handling infrastructure, scaling, and optimization automatically. Compared to using Azure directly, they are easier and faster to use, but developers have less control and flexibility.

---

## Cloud Concepts Question 5

The cloud is probably not the right choice for extremely small personal projects or for systems that require complete physical control and strict security requirements.

---

## Azure Basics Question 1

An Azure subscription controls billing and access to cloud services. A resource group is a container for organizing related resources. The resource group is personal, while the CTD subscription is shared.

---

## Azure Basics Question 2

Ephemeral means Cloud Shell resets temporary files after the session ends. CTD uses mounted cloud storage to make files persistent.

---

## Azure Basics Question 3

The private SSH key stays on your machine and should never be shared. The public SSH key is uploaded to remote systems because it can safely identify your machine without exposing the private key.

---

## Azure Basics Question 4

```bash
az account show
```

Paste your output below:

```text
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#shkuropatenko.d@gmail.com",
    "type": "user"
  }
}
```

Adding `--output table` formats the output into a cleaner table view that is easier to read.
