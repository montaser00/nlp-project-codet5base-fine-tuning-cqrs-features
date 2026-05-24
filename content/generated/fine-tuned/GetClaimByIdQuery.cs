using AutoMapper;
using AutoMapper.QueryableExtensions;
using Axions.Core.Common.Constants;
using Axions.Core.Common.Data;
using Axions.Core.Common.Exceptions;
using Axions.Core.Common.Extensions;
using Axions.Core.Features.Common.Dtos;
using Axions.Core.Features.Claim.Dtos;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Axions.Core.Features.Claim.Queries.GetClaim;

/// <summary>
/// GetClaimByIdQuery
/// </summary>
public class GetClaimByIdQuery: IRequest<BaseResultDto<ClaimDto>>
{
    /// <summary>
    /// Gets or sets Id.
    /// </summary>
    public long Id { get; set; }
}

/// <summary>
/// Handler for GetClaimByIdQuery.
/// </summary>
public class GetClaimByIdQueryHandler(ApplicationDbContext context, IMapper mapper) : IRequestHandler<GetClaimByIdQuery, BaseResultDto<ClaimDto>>
{
    public async Task<BaseResultDto<ClaimDto>> Handle(GetClaimByIdQuery request, CancellationToken cancellationToken)
    {
        var entity = await context.Claims
            .ProjectTo<ClaimDto>(mapper.ConfigurationProvider)
            .AsNoTracking()
            .SingleOrDefaultAsync(x => x.Id == request.Id, cancellationToken)
            .ConfigureAwait(false) ?? throw new NotFoundErrorException(Messages.NotFound);

        return entity.ResultDto<ClaimDto>();
    }
}
