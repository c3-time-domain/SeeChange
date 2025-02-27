// Namespace, which is the only thing exported

var seechange = {};

// utils

seechange.nullorfixed = function( val, num ) { return val == null ? null : val.toFixed(num); }

export { seechange };
