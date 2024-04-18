import datetime
import operator
from functools import partial
import numpy as np
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy import orm

from astropy.time import Time

from models.base import Base, SeeChangeBase, SmartSession, AutoIDMixin, SpatiallyIndexed
from models.measurements import Measurements

import util.config as config


class Object(Base, AutoIDMixin, SpatiallyIndexed):
    __tablename__ = 'objects'

    name = sa.Column(
        sa.String,
        nullable=False,
        unique=True,
        index=True,
        doc='Name of the object (can be internal nomenclature or external designation, e.g., "SN2017abc")'
    )

    is_test = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc='Boolean flag to indicate if the object is a test object created during testing. '
    )

    is_fake = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc='Boolean flag to indicate if the object is a fake object that has been artificially injected. '
    )

    measurements = orm.relationship(
        Measurements,
        back_populates='object',
        cascade='all, delete-orphan',
        passive_deletes=True,
        lazy='selectin',
        doc='All Measurements related to the object, can include duplicates or bad measurements! '
    )

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.calculate_coordinates()

    def get_measurements_list(
            self,
            prov_hash_list=None,
            radius=2.0,
            thresholds=None,
            mjd_start=None,
            mjd_end=None,
            time_start=None,
            time_end=None,
    ):
        """Filter the measurements associated with this object.

        Parameters
        ----------
        prov_hash_list: list of strings, optional
            The prov_hash_list is used to choose only some measurements, if they have a matching provenance hash.
            This list is ordered such that the first hash is the most preferred, and the last is the least preferred.
            If not given, will default to the most recently added Measurements object's provenance.
        radius: float, optional
            Will use the radius parameter to narrow down to measurements within a certain distance of the object's
            coordinates (can only narrow down measurements that are already associated with the object).
            Default is to grab all pre-associated measurements.
        thresholds: dict, optional
            Provide a dictionary of thresholds to cut on the Measurements object's disqualifier_scores.
            Can provide keys that match the keys of the disqualifier_scores dictionary, in which case the cuts
            will be applied to any Measurements object that has the appropriate score.
            Can also provide a nested dictionary, where the key is the provenance hash, in which case the value
            is a dictionary with keys matching the disqualifier_scores dictionary of those specific Measurements
            that have that provenance (i.e., you can give different thresholds for different provenances).
            The specific provenance thresholds will override the general thresholds.
            Note that if any of the disqualifier scores is not given, then the threshold saved
            in the Measurements object's Provenance will be used (the original threshold).
            If a disqualifier score is given but no corresponding threshold is given, then the cut will not be applied.
            To override an existing threshold, provide a new value but set it to None.
        mjd_start: float, optional
            The minimum MJD to consider. If not given, will default to the earliest MJD.
        mjd_end: float, optional
            The maximum MJD to consider. If not given, will default to the latest MJD.
        time_start: datetime.datetime or ISO date string, optional
            The minimum time to consider. If not given, will default to the earliest time.
        time_end: datetime.datetime or ISO date string, optional
            The maximum time to consider. If not given, will default to the latest time.

        Returns
        -------
        list of Measurements
        """
        # this includes all measurements that are close to the discovery measurement
        # measurements = session.scalars(
        #     sa.select(Measurements).where(Measurements.cone_search(self.ra, self.dec, radius))
        # ).all()

        if time_start is not None and mjd_start is not None:
            raise ValueError('Cannot provide both time_start and mjd_start. ')
        if time_end is not None and mjd_end is not None:
            raise ValueError('Cannot provide both time_end and mjd_end. ')

        if time_start is not None:
            mjd_start = Time(time_start).mjd

        if time_end is not None:
            mjd_end = Time(time_end).mjd

        measurements = []
        if radius is not None:
            for m in self.measurements:  # include only Measurements objects inside the given radius
                delta_ra = np.cos(m.dec * np.pi / 180) * (m.ra - self.ra)
                delta_dec = m.dec - self.dec
                if np.sqrt(delta_ra**2 + delta_dec**2) < radius / 3600:
                    measurements.append(m)

        if thresholds is None:
            thresholds = {}

        if prov_hash_list is None:
            # sort by most recent first
            last_created = max(self.measurements, key=operator.attrgetter('created_at'))
            prov_hash_list = [last_created.proveneance.id]

        passed_measurements = []
        for m in measurements:
            local_thresh = m.provenance.parameters.get('thresholds', {}).copy()  # don't change provenance parameters!
            if m.provenance.id in thresholds:
                new_thresh = thresholds[m.provenance.id]  # specific thresholds for this provenance
            else:
                new_thresh = thresholds  # global thresholds for all provenances

            local_thresh.update(new_thresh)  # override the Measurements object's thresholds with the new ones

            for key, value in local_thresh.items():
                if value is not None and m.disqualifier_scores.get(key, 0.0) >= value:
                    break
            else:
                passed_measurements.append(m)  # only append if all disqualifiers are below the threshold

        # group measurements into a dictionary by their MJD
        measurements_per_mjd = defaultdict(list)
        for m in passed_measurements:
            measurements_per_mjd[m.mjd].append(m)

        for mjd, m_list in measurements_per_mjd.items():
            # check if a measurement matches one of the provenance hashes
            for hash in prov_hash_list:
                best_m = [m for m in m_list if m.provenance.id == hash]
                if len(best_m) > 1:
                    raise ValueError('More than one measurement with the same provenance. ')
                if len(best_m) == 1:
                    measurements_per_mjd[mjd] = best_m[0]  # replace a list with a single Measurements object
                    break  # don't need to keep checking the other hashes
            else:
                # if none of the hashes match, don't have anything on that date
                measurements_per_mjd[mjd] = None

        # remove the missing dates
        output = [m for m in measurements_per_mjd.values() if m is not None]

        # remove measurements before mjd_start
        if mjd_start is not None:
            output = [m for m in output if m.mjd >= mjd_start]

        # remove measurements after mjd_end
        if mjd_end is not None:
            output = [m for m in output if m.mjd <= mjd_end]

        return output

    @staticmethod
    def make_naming_function(format_string):
        """Generate a function that will translate a serial number into a name.

        The possible format specifiers are:
        - <instrument> : the (short) instrument name
        - <yyyy> : the year of the object's creation in four digits
        - <yy> : the year of the object's creation in two digits
        - <mm> : the month of the object's creation in two digits
        - <dd> : the day of the object's creation in two digits
        - <alpha> : a lowercase set of letters, starting with a..z and then aa..az..zz then aaa..zzz, etc.
        - <ALPHA> : an uppercase set of letters, starting with A..Z and then AA..AZ..ZZ then AAA..ZZZ, etc.
        - <number> : a number that increments by one for each object created

        The function takes the Object, that must already have a created_at datetime and numeric (autoincrementing) id.
        The second argument to the function is the ID of the last object from last year/month/day (if using that).
        """
        def name_func(obj, starting_id=0, fmt=''):
            # get the current year, month, and day
            replacements = {}
            replacements['yyyy'] = str(obj.created_at.year)
            replacements['yy'] = f'{obj.created_at.year % 100:02d}'
            replacements['mm'] = f'{obj.created_at.month:02d}'
            replacements['dd'] = f'{obj.created_at.day:02d}'

            replacements['number'] = obj.id - starting_id

            if obj.measurements is None or len(obj.measurements) == 0:
                replacements['instrument'] = 'UNK'
            else:
                replacements['instrument'] = obj.measurements[0].instrument_object.get_short_instrument_name()

            # generate the alpha part
            num_to_letters = replacements['number']
            need_letters = 0
            while num_to_letters >= 1:
                num_to_letters //= 26
                need_letters += 1

            if need_letters == 0:
                need_letters = 1  # avoid the case where the number is 0, and then no letters are given

            alpha = ''
            for i in range(need_letters):
                alpha += chr(((replacements['number']) // 26 ** i) % 26 + ord('a'))

            alpha = alpha[::-1]  # reverse the string
            replacements['alpha'] = alpha

            # generate the ALPHA part
            replacements['ALPHA'] = alpha.upper()

            for key in ['instrument', 'yyyy', 'yy', 'mm', 'dd', 'alpha', 'ALPHA', 'number']:
                fmt = fmt.replace(f'<{key}>', str(replacements[key]))

            return fmt

        return partial(name_func, fmt=format_string)

    @staticmethod
    def get_last_id_for_naming(convention, present_time=None, session=None):
        """Get the ID of the last object before the given date (defaults to now).

        Will query the database for an object with a created_at which is the last before
        the start of this year, month or day (depending on what exists in the naming convention).
        Will return the ID of that object, or 0 if no object exists.

        Parameters
        ----------
        convention: str
            The naming convention that will be used to generate the name.
            Example: SomeText_<instrument>_<yyyy><mm>_<number>
        present_time: datetime.datetime, optional
            The time to use as the present time. Defaults to now.
        session: sqlalchemy.orm.session.Session, optional
            The session to use for the query. If not given, will open a new session
            that will be automatically closed at the end of this function.

        Returns
        -------
        int
            The ID of the last object before the given date.
        """
        if present_time is None:
            present_time = datetime.datetime.utcnow()

        # figure out what the time frame should be:
        if '<yyyy>' in convention:
            start_time = datetime.datetime(present_time.year, 1, 1)
        elif '<mm>' in convention:
            start_time = datetime.datetime(present_time.year, present_time.month, 1)
        elif '<dd>' in convention:
            start_time = datetime.datetime(present_time.year, present_time.month, present_time.day)
        else:
            return 0  # if none of these format specifiers are present, just assume there is no last object

        with SmartSession(session) as session:
            last_obj = session.scalars(
                sa.select(Object).where(Object.created_at < start_time).order_by(Object.created_at.desc())
            ).first()
            if last_obj is None:
                return 0
            return last_obj.id


# add an event listener to catch objects before insert and generate a name for them
@sa.event.listens_for(Object, 'before_insert')
def generate_object_name(mapper, connection, target):
    if target.name is None:
        target.name = 'placeholder'


@sa.event.listens_for(sa.orm.session.Session, 'after_flush_postexec')
def receive_after_flush_postexec(session, flush_context):
    cfg = config.Config.get()
    convention = cfg.value('object_naming_function', '<instrument><yyyy><alpha>')
    naming_func = Object.make_naming_function(convention)
    last_id = Object.get_last_id_for_naming(convention, session=session)

    for obj in session.identity_map.values():
        if isinstance(obj, Object) and (obj.name is None or obj.name == 'placeholder'):
            obj.name = naming_func(obj, last_id)
            # print(f'Object ID: {obj.id} Name: {obj.name}')


if __name__ == '__main__':
    import datetime

    obj = Object()
    obj.created_at = datetime.datetime.utcnow()
    obj.id = 130

    fun = Object.make_naming_function('SeeChange<instrument>_<yyyy><alpha>')
    print(fun(obj))
