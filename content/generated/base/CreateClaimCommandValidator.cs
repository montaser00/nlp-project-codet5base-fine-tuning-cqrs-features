, ReferenceId, ClaimCommandValidator> {

    @Override
	public void validate(ClaimCommand command, ValidationContext validationContext) throws ValidationFailedException{
    	if(command.getClaimId() == null){
        	validationContext.addError("claimId", "The claimId is required.");
        }else if(!claimService.isClaimValid(new Claim(null, null, command.getId(), null)))
			throw new ValidationException("Claim with the given id does not exist.", new Object[]{"id"}, new String[]{},
					new Object[] {String.format("%s is not a valid claim id", command)}) ;
   }
}
