using Axions.Core.Common.Constants;
using Axions.Core.Common.Data;
using FluentValidation;
using Microsoft.EntityFrameworkCore;

namespace Axions.Core.Features.Claims.Commands.UpdateClaim;

/// <summary>
/// Validator for UpdateClaimCommand.
/// </summary>
public class UpdateClaimCommandValidator: AbstractValidator<UpdateClaimCommand>
{
    public UpdateClaimCommandValidator(ApplicationDbContext context)
    {
        RuleFor(x => x.Id)
            .Cascade(CascadeMode.Stop)
            .NotEmpty()
            .WithMessage(Messages.FieldIsRequired)
            .MustAsync(async (claimId, cancellationToken) => await context.Claims.AnyAsync(x => x.Id == claimId, cancellationToken).ConfigureAwait(false))
            .WithMessage(Messages.IdNotFound);

        RuleFor(x => x.Date)
            .NotEmpty()
            .WithMessage(Messages.FieldIsRequired);

        RuleFor(x => x.ReferenceId)
            .NotEmpty()
            .WithMessage(Messages.FieldIsRequired);
    }
}
