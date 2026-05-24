, ClaimType

type ClaimCommand struct {
	Id        int64  `json:"id,omitempty"` // The ID of the claim to be claimed. If not specified, the command will claim the current user's default claim. This field can be used to specify a different claim type for a given user. The claim will be created if it does not already exist. Note that if a claim is already created with the same ID, it will not be updated. To create a new claim, use the `create` command. For more information about claim types, see [Claim Types](https://developer.github.com/v3/repos/claims/#create-a-claim-type) and [Claims on GitHub](http://docs.ghc.org/ce/rest/reference/api/organizations.teams/teams/#get-all-teams-for-an-organization-and-team).
}
