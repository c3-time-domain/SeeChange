## Pipeline in depth

TBA.

In the meantime, some details:

### Image positions

Everywhere internally in the code (except where necessary for conversion to and from file formats), we store x and y positions on images according to the `astropy` standard.  That is:

* The center of the lower-left pixel is `(0.0, 0.0)`.  (This is a somewhat perverse standard.  It would make sense if an astronomical image were sampling the distribution of light at points, but it's not; it's integrating in squares.  This means that the pixel coordinates of an image that is sixe `(nx,ny)` cover light on the detector plane in the range `(-0.5–(nx-0.5), -0.5–(ny-0.5)`.  Choosing half-pixel-positions to be centers of pixels would have made more sense, but, alas, integer pixel positions being the center of pixels is a convention that is long with us, so we're stuck with it.)
* Images are stored in Y-major format.  This means that in a 1-d array of pixel values, incrementing the index of the array increases the x-coordinate by one, except at the end of the row where it increases the y-coordinate by one and resets x to 0.  2d Numpy arrays are stored, by default, in "first index major" order.  Numpy calls this "C" order, and documents it as row-major, assuming you are indexing 2d arrays as [row, column].  This means that to extract the value of a pixel at `(x,y)` in 2d numpy array `im`, you need to do `im[y,x]`.

The FITS file format comes form an ancient human civilization that spoke FORTRAN.  While it also stores images in Y-major format, it defines the center of the lower-left pizel as being at (1.0, 1.0).  This means that when you look at WCSes in headers, or when you look at pixels using FITS coordinates (with tools like `ds9`), image coordinates will be off by one from our standard.  Fortunately, `astropy` converts all of this when reading and writing FITS images and WCSes, so for the most part we don't have to worry about it.  However, we _do_ have to worry about it when reading and writing files cerated by Sextractor.

### Database assumptions

If adding new tables, or new columns to existing tables, follow these conventions:

* Each table should have a primary key named `_id`.  (There may be a couple of existing tables that violate this, but new tables should not.)  The model should probably have a property `id` that returns the value of `_id`.  (If your model includes `UUIDMixin`, this just happens.)

* Don't use `default=`, use `server_default=`.  (Reason: so that if we interact with the database outside of the context of SQLAlchemy, the defaults will still apply.)

* Never set a default on a column that is nullable.  If the column is nullable, it should always default to null.  (`models/base.py::SeeChangeBase.insert` et al. (really in `_get_cols_and_vals_for_insert`) makes this assumption.)

### UUIDs as primary keys

If you have asked the question "why are you using UUIDs instead of integers as primary keys", this section is for you.  If you don't care, skip it.

You can find long debates and flamewars on the Internet about using big integers vs. UUIDs as primary keys. The advantages of big integers include:

* Less space used (64-bit vs. 128-bit). (64-bits is plenty of room for what we need.)
* Faster index inserting.
* Clustered indexes. (This is not usually relevant to us. If you're likely to want to pull out groups of rows of a table that were all inserted at the same time, it's a bit more efficient using something sorted like integers rather than something random like UUIDs. Most of the time, this isn't relevant to us; one exception is that we will sometimes want to pull out all measurements from a single subtraction, and those will all have been submitted together.)

Despite these disadvantages, UUIDs offer some advantages, which ultimately end up winning out. They all stem from the fact that you can generate unique primary keys without having to contact the database. This allows us, for example, to build up a collection of objects including foreign keys to each other, and save them all to the database at the end. With auto-generating primary keys, we wouldn't be able to set the foreign keys until we'd saved the referenced object to the database, so that its id was generated. (SQLAlchemy gets around this with object relationships, but object relationships in SA caused us so many headaches that we stopped using them; see below.)  It also allows us to do things like cache objects that we later load into the database, without worrying that the cached object's id (and references among multiple cached objects) will be inconsistent with the state of the database counters.

(Note that there are [performance reasons to prefer UUID7 over UUID4](https://ardentperf.com/2024/02/03/uuid-benchmark-war/), but at the moment we're using v4 UUIDs because the python uuid library doesn't support V7.  If at some future time it does, it might be worth changing.)

### Use of SQLAlchemy

This is for developers working on the pipeline; users can ignore this section.

SQLAlchemy provides a siren song: you can access all of your database as python objects without having to muck about with SQL!  Unfortunately, just like the siren song of Greek myth, if you listen to it, you're likely to drown. One of the primary authors of this pipeline has come around to the view, which you can find in the various flamewars about ORMs (Object Relational Mappers) on the net, that ORMs make easy things easy, and make complicated things impossible.

If you're working in a situation where you can create a single SQLAlchemy database session, hold that session open, and keep all of your objects attached to that session, then SQLAlchemy will probably work more or less as intended. (You will still end up with the usual ORM problem of not really knowing what your database accesses are, and whether you're unconsciously constructing highly inefficient queries.)  However, for this code base, that's not an option. We have long-running processes (subtracting an searching an image takes a minute or two in the best case), and we run lots of them at once (tens of processes for a single exposure to cover all chips, and then multiple nodes doing different exposures at once). The result is that we would end up with hundreds of connections to the database held open, most of them sitting idle most of the time. Database connections are a finite resource; while you can configure your database to allow lots of them, you may not always have the freedom to do that, and it's also wasteful. When you're doing seconds or minutes (as opposed to hundredths or tenths of seconds) of computation between database accesses, the overhead of creating new connections becomes relatively small, and not worth the cost to the database of keeping all those connections open. In a pipeline like this, much better practice is to open a connection to the database when you need it and hold it open only as long as you need it. With SQLAlchemy, that means that you end up having to shuffle objects between sessions as you make new sessions for new connections.  This undermines a lot of what SQLAlchemy does to hide you from SQL, and can rapidly end up with a nightmare of detached instance errors, unique constraint violations, and very messy "merge" operations. You can work around them, and for a long time we did, but the result was long complicated bits of code to deal with merging of objects and related objects, and "eager loading" meaning that all relationships between objects got loaded from the database even if you didn't need them, which is (potentially very) inefficient. (What's more, we regularly ran into issues where debugging the code was challenging because we got some SQLAlchemy error, and we had to try to track down which object we'd failed to merge to the session properly. So much time was lost to this.)

We still use SQLAlchemy, but have tried to avoid most of its dysfunctionality in cases where you don't keep a single session in which all your objects live. To this end, when defining SQLAlchemy models, follow these rules:

* Do _not_ define any relationships. These are the things that lead to most of the mysterious SQLAlchemy errors we got, as it tried to automatically load things but then became confused when objects weren't attached to sessions. They also led to our having to be very careful to make sure all kinds of things were merged before trying to commit stuff to the database. (It turned out that the manual code we had to write to load the related objects ourselves was much less messy than all the merging code.)  Of course you can still have foreign keys between objects, just don't define something that SQLAlchemy calls a "relationship", because that's where the complexity arises.

* Do not use any association proxies. These are just relationships without the word "relationship" in the name.

* Always get your SQLAlchemy sessions inside a the models.base.SmartSession context manager (i.e. `with SmartSession() as session`). Assuming you're passing no arguments to SmartSession() (which should usually, but not always, be the case--- you can find examples of its use in the current code), then this will help in not holding database connections open for a long time.

* Don't hold sessions open. Make sure that you only put inside the `with SmartSession()` block the actual code you need to access the database, and don't put any long calculations inside that `with` block.  (If you make a function call that also accesses the database inside this call, you may end up with a deadlock, as some of the library code locks tables.)  Also, __never save the session variable to a member of an object or anything else__. That could prevent the session from really going out of scope, and stop SA from properly garbage collecting it.  (Maybe.)

You may ask at this point, why use SQLAlchemy at all?  You've taken away a lot of what it does for you (though, of course, that also means we have removed the costs of letting it do that), and now have it as more or less a thin layer in front of SQL. The reasons are threefold:

* First, and primarily, `alembic` is a nice migration manager, and it depends on SQLAlchemy.

* Some of the syntactic sugar from SQLAlchemy (e.g. `objects=session.query(Class).filter(Class.property==value).all()`) are probably nicer for most people to write than embedding SQL statements.

Of course, we still end up with some SQLAlchemy weirdness, because it _really_ wants you to just leave objects attached to sessions, so some very basic operations sometimes still end up screwing things up.  You will find a few workarounds (with irritated comments ahead of them) in the code that deal with this.  We also ended up writing explicit SQL in the code that inserts and update objects in the database (see `base.py::SeeChangeBase.insert()` and `upsert()`), as it turns out _any_ time you use the word `merge` in association with SQLAlchmey, you're probably setting yourself up for a world of hurt.  Except for several places in tests, we've managed to get ourselves down to a single call to the SQLAlchmey `merge` method, in `base.py::HasbitFlagBadness.update_downstream_badness()`.)

We also suffer because (apparently) there is no way to explicitly and immediately close the SQLAlchmey connection to the database; it seems to rely on garbage collection to actually close sessions that have been marked as closed and invalidated.  While this works most of the time, occasionally (and unreproducibly) a session lingers in an idle transaction.  This caused troubles in a few cases where we wanted to use table locks to deal with race conditions, as there would be a database deadlock.  We've worked around it by reducing the number of table locks as much as possible (which, frankly, is a good idea anyway).  To really get around this problem, unless there's a way to force SQLAlchmey to really close a connection when you tell it to close, we'd probably have to refactor the code to not use SQLAlchmey at all, which would be another gigantic effort.  (Using a `rollback()` call on the session would close the idle transaction, but unfortunately that has the side effect of making every object attached to that session unusable thereafter.)
