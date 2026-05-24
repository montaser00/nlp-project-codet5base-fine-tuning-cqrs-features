using Axions.Core.Common.Constants;
using FluentValidation;

namespace Axions.Core.Features.Claims.Commands.GetClaims;

/// <summary>
/// Validator for GetClaimsCommand.
/// </summary>
public class GetClaimsCommandValidator: AbstractValidator<GetClaimsCommand>
{
    public GetClaimsCommandValidator()
    {
        RuleFor(x => x.Date)
            .NotEmpty()
            .WithMessage(Messages.FieldIsRequired);
    }
}
