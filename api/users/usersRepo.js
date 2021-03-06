module.exports.name = 'usersRepo'
module.exports.dependencies = ['db', 'User', '@polyn/blueprint']
module.exports.factory = (db, User, _blueprint) => {
  'use strict'

  const { is } = _blueprint
  const collection = db.collection(User.db.collection)

  User.db.indexes.forEach(index => {
    collection.createIndex(index.keys, index.options)
  })

  /*
    // Create a user
    */
  const create = (payload) => {
    return new Promise((resolve, reject) => {
      if (is.not.object(payload)) {
        return reject(new Error('A payload is required to create a User'))
      }

      collection.insertOne(payload, (err, res) => {
        if (err) {
          return reject(err)
        }

        return resolve(res)
      })
    })
  }

  /*
    // Get a single user
    */
  const get = (email) => {
    return new Promise((resolve, reject) => {
      if (is.not.string(email)) {
        return reject(new Error('An email is required to get a User'))
      }

      collection.find({ email })
        .limit(1)
        .next((err, doc) => {
          if (err) {
            return reject(err)
          }

          if (doc) {
            return resolve(doc)
          } else {
            return resolve()
          }
        })
    })
  }

  return { create, get }
}
