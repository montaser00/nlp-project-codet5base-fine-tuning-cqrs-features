using Axions.Core.Common.Constants;
using Axions.Core.Common.Data;
using Axions.Core.Common.Exceptions;
using Axions.Core.Features.Common.Dtos;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Axions.Core.Features.Claims.Commands.CreateClaim;

/// <summary>
/// Command for creating Claim.
/// </summary>
public class CreateClaimCommand: IRequest<BaseResultDto<long>>
{
    /// <summary>
    /// Gets or sets Date.
    /// </summary>
    public DateTimeOffset? Date { get; set; }

    /// <summary>
    /// Gets or sets ReferenceId.
    /// </summary>
    public long? ReferenceId { get; set; }
}

/// <summary>
/// CreateClaimCommand Handler.
/// </summary>
public class CreateClaimCommandHandler(ApplicationDbContext context) : IRequestHandler<CreateClaimCommand, BaseResultDto<long>>
{
    #region Handler

    public async Task<BaseResultDto<long>> Handle(CreateClaimCommand request, CancellationToken cancellationToken)
    {
        var entity = new Claim
        {
            Date = request.Date,
            ReferenceId = request.ReferenceId,
        };

        context.Claims.Add(entity);
        var result = await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

        if (result < 0)
            throw new InternalApplicationErrorException(Messages.SaveFailed);

        return new BaseResultDto<long>(entity.Id);
    }

    #endregion
}
