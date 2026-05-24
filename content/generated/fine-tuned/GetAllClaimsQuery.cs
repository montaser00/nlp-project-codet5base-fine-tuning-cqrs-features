using AutoMapper;
using AutoMapper.QueryableExtensions;
using Axions.Core.Common.Constants;
using Axions.Core.Common.Data;
using Axions.Core.Common.Exceptions;
using Axions.Core.Common.Extensions;
using Axions.Core.Features.Common.Dtos;
using Axions.Core.Features.Claims.Dtos;
using MediatR;
using Microsoft.EntityFrameworkCore;

namespace Axions.Core.Features.Claims.Queries.GetClaims;

/// <summary>
/// GetsClaimsQuery
/// </summary>
public class GetClaimsQuery: IRequest<BaseResultDto<PaginatedResult<ClaimDto>>>
{
    /// <summary>
    /// Gets or sets Date.
    /// </summary>
    public DateTime? Date { get; set; }
}

/// <summary>
/// Handler for GetClaimsQuery.
/// </summary>
public class GetClaimQueryHandler(ApplicationDbContext context, IMapper mapper) : IRequestHandler<GetClaimsQuery, BaseResultDto<PaginatedResult<ClaimDto>>>
{
    public async Task<BaseResultDto<PaginatedResult<ClaimDto>>> Handle(GetClaimsQuery request, CancellationToken cancellationToken)
    {
        var entities = await context.Claims
            .ProjectTo<ClaimDto>(mapper.ConfigurationProvider)
            .AsNoTracking()
            .SingleOrDefaultAsync(x => x.Date == request.Date, cancellationToken)
            .ConfigureAwait(false) ?? throw new NotFoundErrorException(Messages.NotFound);

        return entities.ResultDto<PaginatedResult<ClaimDto>>();
    }
}
