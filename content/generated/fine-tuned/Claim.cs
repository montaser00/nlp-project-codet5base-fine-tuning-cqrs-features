using Axions.Core.Common.Constants;
using Axions.Core.Common.Data;
using Axions.Core.Common.Exceptions;
using Axions.Core.Features.Common.Dtos;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Axions.Core.Features.Claims.Entities;

public class Claim: BaseEntity, IMapFrom<ClaimDto>
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
/// Entity for Claim.
/// </summary>
public class ClaimEntity: BaseEntity, IMapFrom<ClaimDto>
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
/// Handler for Claim.
/// </summary>
public class ClaimHandler(ApplicationDbContext context) : IRequestHandler<Claim, ClaimDto>
{
    #region Handler

    public async Task<ClaimDto> Handle(Claim request, CancellationToken cancellationToken)
    {
        var entity = await context.Claims
            .FirstOrDefaultAsync(x => x.Id == request.Id, cancellationToken)
            .ConfigureAwait(false) ?? throw new NotFoundErrorException(Messages.NotFound);

        return entity.ToDto();
    }

    #endregion
}
