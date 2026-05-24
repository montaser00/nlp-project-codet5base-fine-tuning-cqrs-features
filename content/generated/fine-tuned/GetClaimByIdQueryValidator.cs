using Axions.Core.Common.Constants;
using FluentValidation;

namespace Axions.Core.Features.Claims.Commands.GetClaim;

/// <summary>
/// Validator for GetClaimByIdCommand.
/// </summary>
public class GetClaimByIdCommandValidator: AbstractValidator<GetClaimByIdCommand>
{
    public GetClaimByIdCommandValidator()
    {
        RuleFor(x => x.Id)
            .NotEmpty()
            .WithMessage(Messages.FieldIsRequired);
    }
}
