## Troubleshooting SQLAlchemy

Here is a growing list of common issues and solutions for SQLAlchemy.

#### Adding this object again causes a unique constraint violation

This is a common issue when you are trying to add an object to the session that is already on the DB. 
Instead, use merge, and make sure to assign the merged object to a variable (often with the same name) 
and keep using that. There's no real advantage to using `session.add()` over `session.merge()`. 

Example: 

```python
obj = session.merge(obj)
```

#### Related objects get added to the session (and database) when they are not supposed to

This is a hard one, where a complex web of relationships is causing SQLAlchemy to add objects to the session 
when they are not supposed to.
This happens when you `session.merge()` an object, not just on `session.add()`. 
This is especially tricky when you are trying to delete a parent, so you merge it first, 
and then you end up adding the children instead. 
Usually the relationship will merge and then delete the children using cascades, 
but some complex relationships may not work that way. 
If you notice things are getting added when they shouldn't, check the session state before committing/flushing. 

The places to look are: 
```python
session.identity_map.keys()
session.new
session.dirty
session.deleted
```

If unwanted objects appear there, try to `session.expunge()` them before committing, or if they are persistent, 
you may need to `session.delete()` them instead. 

#### Double adding a related object through cascades

Sometimes when a child is merged (or added) into a session, the parent is not automatically added. 
Then, when the parent is added to the session on its own, it gets added as a new object, that can trigger
unique violations (or, worse, just add duplicates). 

The root of this problem is that the child object is merged without the parent. 
Remember that a merged object is a new copy of the original, only connected to the session. 
If you don't cascade the merge to the parent, you can't just assign the parent to the new object. 
The parent object still keeps a reference to the old child object, and that one is not on the session. 
Instead, make sure the merged child is assigned a merged parent, and that the parent is related 

#### Cannot access related children when parent is not in the session

This happens when a parent object is not in the session, but you want to access its children.
The error message is usually something like this: 

```
sqlalchemy.orm.exc.DetachedInstanceError: Parent instance <Parent at 0x7f7f7f7f7f7f> is not bound to a Session; 
lazy load operation of attribute 'children' cannot proceed
```

This happens under three possible circumstances. 
1. The relationship is lazy loaded (which we generally try to avoid). 
   Check the relationship definition has `lazy='selectin'`.
2. The parent object was loaded as a related object itself, and that loading did not recursively load the children. 
   Most objects will recursively load related objects of related objects, but in some cases this doesn't work, 
   in particular when there's a many-to-many relationship via an association table (e.g., Provenance.upstreams). 
   This is fixed by setting the `join_depth=1` or higher, as documented 
   [here](https://docs.sqlalchemy.org/en/20/orm/self_referential.html#configuring-self-referential-eager-loading)
3. The session has rolled back, or committed (this option only if you've changed to expire_on_commit=True). 
   We usually have expire_on_commit=False, so that objects do not get expired when the session is committed.
   However, when the session is rolled back, all objects are expired. That means you cannot use related objects, 
   or even regular attributes, after a rollback. In most cases, a rollback is due to some crash, so having some 
   errors accessing attributes/relationships while handling exceptions and "gracefully" exiting the program is expected, 
   and doesn't require too much attention. If, however, you explicitly called a rollback, you should expect to have 
   expired objects, and should go ahead and `session.refresh()` all the objects you need to use.

#### Parent not in session, update along children is not updated in the database (Warning only)

This is a warning that tells you that even though you added / deleted a child object, 
the relationship cannot automatically update the object in the database, because the parent 
is not connected to a session. 

This is sometimes important but a lot of times meaningless. For example, if you deleted Parent, 
and then go on to remove the children from it, it makes little difference that the relationship 
is no longer emitting SQL changes, because the parent is going to be deleted anyway.


#### `When initializing mapper Mapper[...], expression '...' failed to locate a name `

This happens when a related object class is not imported when the relationship needs to be instantiated. 

When two classes, A and B, are related to each other, we would see a definition like this: 

```python
class A(Base):
    __tablename__ = 'a'
    id = Column(Integer, primary_key=True)
    b_id = Column(Integer, ForeignKey('b.id'))
    b = relationship('B')

class B(Base):
    __tablename__ = 'b'
    id = Column(Integer, primary_key=True)
    a_id = Column(Integer, ForeignKey('a.id'))
    a = relationship('A')
```

Notice that the `relationship` function is called with a string argument. 
This is because the class `B` is not defined yet when the class `A` is defined.
This solves a "chicken and egg" problem, by making a promise to the mapper that 
when the relationships are instatiated, both classes will have been imported. 

If some of the related objects are on a different file (module) and that file 
is not imported by any of the code you are running, you will get the error above.

This usually happens on scripts and parallel pipelines that only use a subset of the classes.
To fix this, simply import the missing class module at the beginning of the script. 


#### Changing the primary key of an object causes update instead of new object

For objects that don't have an auto-incrementing primary key (e.g., Provenance),
the user is in control of the value that goes into the primary key. 
Sometimes, the user changes this value, e.g., when a Provenance gets new parameters
and the `update_id()` method is called. 

If the object is already in the session, and the primary key is changed, SQLAlchemy
will update the object in the database, instead of creating a new one.
This will remove the old object and may cause problems with objects that relate to 
that row in the table. 

Make sure to detach your object, or make a brand new one and copy properties over 
to the new instance before merging it back into the session as a new object. 


#### Deadlocks when querying the database

This can occur when an internal session is querying the same objects 
that an external session is using. 
In general, you should not be opening an internal session when a different one is open, 
instead, pass the session as an argument into the lower scope so all functions use the same session.

If the app freezes, check for a deadlock: 
Go into the DB and do `select * from pg_locks;` to see if there are many locks.

Sometimes using `SELECT pg_cancel_backend(pid) FROM pg_locks; ` will free the lock. 
Otherwise, try to restart the psql service. 
