# Data Card: Axions Radius — CQRS Code Generation Dataset

## Dataset Overview

| Field | Details |
|---|---|
| **Name** | Axions Radius CQRS Dataset |
| **Source** | Personal Project — Axions Radius |
| **Created By** | More samples generated using Claude, following the same conventions and code style as the original project |
| **Size** | 833 records |
| **License** | None |

## Description

A supervised instruction-following dataset of input/output pairs for generating C# source code components in a **CQRS (Command Query Responsibility Segregation), Feature-Based, Clean Architecture** style — mirroring the codebase of the Axions Radius project.

Each record contains a natural-language prompt describing a code component to generate, and the corresponding C# source code as the expected output.

## Programming Language & Architecture

- **Language:** C#
- **Pattern:** CQRS via MediatR
- **Architecture:** Feature-Based Clean Architecture
- **Key Libraries:** MediatR, FluentValidation, Entity Framework Core

## Component Types

The dataset covers five categories of CQRS components:

- **Commands** — Create, Update, and Delete command classes with their MediatR handlers
- **Queries** — Query classes with handlers returning result DTOs
- **Validators** — FluentValidation validators for commands and queries
- **DTOs** — Data Transfer Objects used in query responses and command results
- **Models / Entities** — Domain entity classes

## Input / Output Format

Each record is a JSON object:

```json
{
  "input": "<natural language instruction>",
  "output": "<generated C# source code>"
}
```

**Input format:**
```
Create CQRS component.
Feature Name: <FeatureName> Type: <ComponentType> Attributes: <Attr1>, <Attr2>, ...
```

**Output format:** Complete, compilable C# file including namespace declarations, XML doc comments, class definition, and handler logic.

## Conventions & Code Style

All samples follow these consistent conventions:

- Namespace pattern: `Axions.Core.Features.<Feature>.<Commands|Queries>.<ActionFeature>`
- XML `<summary>` doc comments on every class and property
- Properties use nullable types (`string?`, `bool?`, `long?`, `DateTimeOffset?`)
- Handlers use primary constructor injection (C# 12 style)
- `ConfigureAwait(false)` on all async calls
- Soft-delete pattern (`IsDeleted = true`) for Delete commands
- FluentValidation with `Messages.*` constants for error messages
- Update validators use async `MustAsync` to check entity existence in DB
- `BaseResultDto<T>` as the standard return wrapper
- `context.HasChanges()` guard before saving on Update operations

## Example Record

**Input:**
```
Create CQRS component.
Feature Name: Customers Type: Create Command Attributes: FullName, Mobile, Email, Status, RouterId
```

**Output:**
```csharp
using Axions.Core.Common.Constants;
using Axions.Core.Common.Data;
using Axions.Core.Common.Exceptions;
using Axions.Core.Features.Common.Dtos;
using Axions.Core.Features.Customers.Entities;
using MediatR;

namespace Axions.Core.Features.Customers.Commands.CreateCustomer;

/// <summary>
/// Command for creating Customer.
/// </summary>
public class CreateCustomerCommand: IRequest<BaseResultDto<long>>
{
    /// <summary>
    /// Gets or sets FullName.
    /// </summary>
    public string? FullName { get; set; }

    /// <summary>
    /// Gets or sets Mobile.
    /// </summary>
    public string? Mobile { get; set; }

    /// <summary>
    /// Gets or sets Email.
    /// </summary>
    public string? Email { get; set; }

    /// <summary>
    /// Gets or sets Status.
    /// </summary>
    public string? Status { get; set; }

    /// <summary>
    /// Gets or sets RouterId.
    /// </summary>
    public long? RouterId { get; set; }
}

/// <summary>
/// Create Customer Command Handler.
/// </summary>
public class CreateCustomerCommandHandler(ApplicationDbContext context) : IRequestHandler<CreateCustomerCommand, BaseResultDto<long>>
{
    #region Handler

    public async Task<BaseResultDto<long>> Handle(CreateCustomerCommand request, CancellationToken cancellationToken)
    {
        var entity = new Customer
        {
            FullName = request.FullName,
            Mobile = request.Mobile,
            Email = request.Email,
            Status = request.Status,
            RouterId = request.RouterId,
        };

        context.Customers.Add(entity);
        var result = await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

        if (result < 0)
            throw new InternalApplicationErrorException(Messages.SaveFailed);

        return new BaseResultDto<long>(entity.Id);
    }

    #endregion
}
```
