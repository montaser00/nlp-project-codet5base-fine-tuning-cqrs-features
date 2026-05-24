.
 *
 */
class ClaimEntity {

    constructor(id, date, referenceId){
        this.id = id;
		this.date = new Date(Date.parse(date) || Date.now()).toISOString().replace(/T/g, ' ').split(' ')[0];
    	if(!referenceId)
			throw new Error('ReferenceId is required.');
        	else if(typeof(ReferenceEntity) === 'undefined')
            throw new ReferenceEntity('The ReferenceId must be an instance of ' + typeof(ClaimEntity).name + '.'); 
        else 
            this._referenceEntity = referenceEntity; 
    }
}