/* jshint bitwise: false */
module.exports.name = 'common-idFactory'
module.exports.singleton = true
module.exports.dependencies = ['ObjectID', 'logger']
module.exports.factory = function (ObjectID, logger) {
  'use strict'

  var idFactory = {
    /*
        // Generates a new BSON ObjectID, or converts a string into one
        // @param uid (optional)(string): when a uid is supplied, it will
        //      attempt to convert that string into a BSON ObjectID
        // @returns (BSON ObjectID): the unique identifier
        */
    makeObjectId: makeObjectId,
    /*
        // Generates a new unqiue identifer
        // @param length (optional)(int)(default = 8): when supplied the
        //      generated uid will be the length desired
        // @returns (string): the unique identifier
        */
    makeUid: makeUid,
    /*
        // Generates a 32 character Universally Unique Identifier
        */
    makeUuid: makeUuid
  }

  function createRandomString (templateString) {
    return templateString.replace(/[xy]/g, function (c) {
      const r = Math.random() * 16 | 0
      const v = c === 'x' ? r : r & 3 | 8

      return v.toString(16)
    })
  }

  function makeObjectId (uid) {
    if (!uid) {
      var id = new ObjectID()
      return id
    } else if (ObjectID.isValid(uid)) {
      return new ObjectID(uid)
    } else {
      const err = new Error(`Invalid ObjectID: ${uid.toString()}`)
      err.isException = true
      logger.error(err)

      return err
    }
  }

  function makeUid (length) {
    var template

    length = (typeof length === 'number' && length > 0 && length) || 8
    template = new Array(length + 1).join('x')

    return createRandomString(template)
  }

  function makeUuid () {
    return createRandomString('xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx')
  }

  return idFactory
}
