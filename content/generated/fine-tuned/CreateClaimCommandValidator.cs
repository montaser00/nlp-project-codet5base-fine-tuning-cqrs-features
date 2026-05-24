using Axions.Core.Common.Constants;
using FluentValidation;

namespace Axions.Core.Features.Claims.Commands.CreateClaim;

/// <summary>
/// Validator for CreateClaimCommand.
/// </summary>
public class CreateClaimCommandValidator: AbstractValidator<CreateClaimCommand>
{
    public CreateClaimCommandValidator()
    {
        RuleFor(x => x.Date)
            .NotEmpty()
            .WithMessage(Messages.FieldIsRequired);

        RuleFor(x => x.ReferenceId)
            .NotEmpty()
            .WithMessage(Messages.FieldIsRequired);
    }
}
