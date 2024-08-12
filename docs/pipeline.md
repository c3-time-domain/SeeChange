## Pipeline in depth

TBA

### UUIDs as primary keys

If you have asked the question "why are you using UUIDs instead of integers as primary keys", this section is for you.  If you don't care, skip it.

You can find long debates and flamewars on the Internet about using big integers vs. UUIDs as primary keys. The advantages of big integers include:

* Less space used (64-bit vs. 128-bit). (64-bits is plenty of room for what we need.)
* Faster index inserting.
* Clustered indexes. (This is not usually relevant to us. If you're likely to want to pull out groups of rows of a table that were all inserted at the same time, it's a bit more efficient using something sorted like integers rather than something random like UUIDs. Most of the time, this isn't relevant to us; one exception is that we will sometimes want to pull out all measurements from a single subtraction, and those will all have been submitted together.)

Despite these disadvantages, UUIDs offer some advantages, which ultimately end up winning out. They all stem from the fact that you can generate unique primary keys without having to contact the database. This allows us, for example, to build up a collection of objects including foreign keys to each other, and save them to the database at the end. With auto-generating primary keys, we wouldn't be able to set the foreign keys until we'd saved the referenced object to the databse, so that its id was generated. (SQLAlchemy gets around this with object relationships, but object relationships in SA caused us so many headaches that we stopped using them; see below.)  It also allows us to do things like cache objects that we later load into the database, without worrying that the cached object's id (and references amongst multiple cached objects) will be inconsistent with the state of the database counters.


### Use of SQLAlchemy

This is for developers working on the pipeline; users can ignore this section.

SQLAlchemy provides a siren song: you can access all of your database as python objects without having to muck about with SQL!  Unfortunately, just like the siren song of Greek myth, if you listen to it, you're likely to drown. One of the primary authors of this pipeline has come around to the view that you can find in the various flamewars about ORMS (Object Relational Mappers) on the net that ORMs make easy things easy, and make complicated things impossible.

If you're working in a situation where you can create a single SQLAlchemy database session, hold that session open, and keep all of your objects attached to that session, then SQLAlchemy will probably work more or less as intended. (You will still end up with the usual ORM problem of not really knowing what your database accesses are, and whether you're unconsciously constructing highly inefficient queries.)  However, for this code base, that's not an option. We have long-running processes (subtracting an searching an image takes a minute or two in the best caes), and we run lots of them at once (tens of processes for a single exposure to cover all chips, and then multiple nodes doing differente exposures at once). The result is that we would end up with hundreds of connections to the databse held open, most of them sitting idle most of the time. Database connections are a finite resource; while you can configure your database to allow lots of them, you may not always have the freedom to do that, and it's also wasteful. When you're doing seconds or minutes (as opposed to hundreths of seconds) of computation between database accesses, the overhead of creating new connections becomes small, and not worth the cost to the databse of keeping all those connections open. In a pipeline like this, much better practice is to open a connection to the database when you need it and hold it open only as long as you need it. With SQLAlchemy, that means that you end up having to shuffle objects between sessions as you make new ones. This undermines a lot of what SQLAlchemy does to hide you from SQL, and can rapidly end up with a nightmare of detached instance errors and unique constraint violations. You can work around them, and for a long time we did, but the result was long complicated bits of code to deal with merging of objects and related objects, and "eager loading" meaning that all relationships between objects got loaded from the databse even if you didn't need them, which is inefficient. (What's more, we reguarly ran into issues where debugging the code was challenging because we got some SQLAlchemy error, and we had to try to track down which object we'd failed to merge to the session properly. So much time was lost to this.)

We still use SQLAlchemy, but have tried to avoid most of its dysfunctionality in cases where you don't keep a single session in which all your objects live. To this end, when defining SQLAlchemy models, follow these rules:

* Do _not_ define any relationships. These are the things that lead to most of the mysterious SQLAlchemy errors we got, as it tried to automatically load things but then became confused when objects weren't attached to sessions. They also led to our having to be very careful to make sure all kinds of things were merged before trying to commit stuff to the database. (It turned out that the manual code we had to write to load the related objects ourselves was much less messy than all the merging code.)

* Do not use any association proxies. These are just relationships without the word "relationship" in the name.

* Always get your SQLAlchemy sessions inside a the models.base.SmartSession context manager (i.e. `with SmartSession() as session`). Assuming you're passing no arguments to SmartSession() (which should usually, but not always, be the case--- you can find exampels of its use in the current code), then this will help in not holding database connections open for a long time.

* Don't hold sessions open. Make sure that you only put inside the `with SmartSession()` block the actual code you need to access the databse, and don't put any long calculations inside that `with` block

You may ask at this point, why use SQLAlchemy at all?  You've taken away a lot of what it does for you (though, of course, that means we have removed the costs of letting it do that), and now have it as more or less a thin layer in front of SQL. The reasons are threefold:

* First, and primarily, `alembic` is a nice migration manager, and it depends on SQLAlchemy.

* It still does save us the need to write our own code to translate object fields into INSERT or UPDATE statements, and to parse the result of SELECT statements to populate object fields. As long as all the columns you define are simple ones (i.e. not relationships), then what SQLAlchemy is doing is pretty straightforward, and the actual database queries that get run are less likely to be totally obscured from you.

* Some of the syntatic sugar from SQLAlchemy (e.g. `objects=session.query(Class).filter(Class.property==value).all()`) are probably nicer for most people to write than embedding SQL statements.