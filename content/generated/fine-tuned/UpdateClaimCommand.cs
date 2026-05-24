using Axions.Core.Common.Constants;
using Axions.Core.Common.Data;
using Axions.Core.Common.Exceptions;
using Axions.Core.Features.Common.Dtos;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Axions.Core.Features.Claims.Commands.UpdateClaim;

/// <summary>
/// UpdateClaimCommand.
/// </summary>
public class UpdateClaimCommand: IRequest<BaseResultDto<long>>
{
    /// <summary>
    /// Gets or sets Id.
    /// </summary>
    public long Id { get; set; }

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
/// UpdateClaimCommand Handler.
/// </summary>
public class UpdateClaimCommandHandler(ApplicationDbContext context) : IRequestHandler<UpdateClaimCommand, BaseResultDto<long>>
{
    #region Handler

    public async Task<BaseResultDto<long>> Handle(UpdateClaimCommand request, CancellationToken cancellationToken)
    {
        var entity = await context.Claims
            .FirstOrDefaultAsync(x => x.Id == request.Id, cancellationToken)
            .ConfigureAwait(false) ?? throw new NotFoundErrorException(Messages.NotFound);

        entity.Date = request.Date;
        entity.ReferenceId = request.ReferenceId;

        context.Claims.Update(entity);

        if (context.HasChanges())
        {
            var result = await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
            if (result < 0)
                throw new InternalApplicationErrorException(Messages.SaveFailed);
        }

        return new BaseResultDto<long>(entity.Id);
    }

    #endregion
}
