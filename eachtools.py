# === "each" containers (list-like containers that distribute most operations over each of their members) ---

__author__ = "Justin C. Fisher"

#/*
# * Copyright 2022-2023. Justin Fisher.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

from typing import TypeVar, Generic, Union, Iterable, overload, Tuple, Sequence, Collection, Sized, Mapping, \
                   MutableMapping, Callable, Set, Any
import itertools
from math import trunc, floor, ceil

T = TypeVar('T')                          # Used to indicate one-off type relations
MemberType = TypeVar('MemberType')        # Used to indicate the type of members/values in an EachContainer
SubmemberType = TypeVar('SubmemberType')  # Used to indicate type of submembers contained within members
PluralIndexType = Union[slice, Iterable]  # E[i] will return another EachContainer if i is of PluralIndexType
OtherType = TypeVar('OtherType')          # Used to indicate the type of other when broadcasting other to self.
KeyType = TypeVar('KeyType')              # Used to indicate the type of keys in an EachDictionary
OutputType = TypeVar('OutputType')        # Used to indicate the type of outputs produced by an EachFunction


# Some guiding principles:
#  each(x) is a version of x that will generally try to do distributed operations, matching members by index/key, and
#  broadcasting scalars and size-1 singletons.  Mappings like each(D) use their keys as indices, so will typically
#  engage in vectorized ops with other mappings with the same keys, though they can directly join non-mappings in
#  vectorized operations by having the relevant integers as keys, and each(D).key and each(D).value are treated
#  like 0-indexed sequences, allowing each(D) to indirectly engage in vectorized ops with sequences.
#  Non-mapping containers and iterables are presumed to be 0-indexed in their iteration order, even if they don't
#  define __getitem__, or even if they have a custom getitem that would return other values.  However, when explicitly
#  getting items using one or multiple indices, e.g. with E[0] or E[[0,1,2]], their actual __getitem__ will be called,
#  or an error will be raised if they lack one.  Non-string iterables are generally treated as containers to be
#  be distributed in ops.  To delay distribution, encapsulate an iterable in [] to make it a singleton whose
#  singular content will be broadcast in the next operation it encounters.


# === Broadcasting ===

def as_sized_iterable(source) -> Collection:
    """Returns a version of source that is sized and iterable (i.e. that is a Python "Collection").
       If source is a scalar (string or non-iterable), it is returned, wrapped in a tuple.
       If source is a non-string iterable with a computable length, it is returned as is.
       If source is iterable without a computable length (e.g a generator), it is spieled into a tuple and returned."""
    if isinstance(source, str): return (source,)
    try:
        length = len(source)
        # TODO should we worry about the possibility of sized but non-iterable sources?
        return source
    except:
        return tuple(source) if isinstance(source, Iterable) else (source,)

def broadcast_to_length(source, n):
    """Returns an iterator of n items drawn from source.  If source has that length, it will be iterated.
       If source has length 1, its one member will be repeated.  Otherwise source itself will be repeated."""
    # TODO this won't itself unspiel an Iterator and check its length.  Is that fine?
    try:
        length = len(source)
    except:
        length = None
    if length == n:
        return iter(source)
    if length == 1:
        return itertools.repeat(next(source), n)
    return itertools.repeat(source, n)

def broadcast_to_indices(source:Collection, n: int, indices: Iterable) -> Iterable:
    """Returns an iterator of n items drawn from source.
         If source is a Mapping, items in it will be looked up by the given indices.
         Otherwise, if source has length 1, its one member will be repeated n times.
         Otherwise, if indices is not a range object, items will be looked up by the given indices.
         Otherwise, indices are a range object, but if source has insufficient length, a ValueError will be raised.
         Otherwise source itself will be iterated."""
    if isinstance(source, Mapping):
        return (source[i] for i in indices)
    length = len(source)
    if length == 1:
        return itertools.repeat(next(iter(source)), n)
    if not isinstance(indices, range):  # TODO will this case ever trigger?
        return (source[i] for i in indices)  # will raise error if source lacks __getitem__
    if length < n:
        raise ValueError(f"{source} has length {length}, so cannot be broadcast to length {n}.")
    if isinstance(source, EachContainer):
        return source.values()  # will iterate the values, perhaps wrapping in each() due for nested settings
    # TODO this presumes that simply iterating source is equivalent to accessing indices 0..(n-1).  Generalize?
    return source


# @overload  # with_matched_version_of(self, scalar) returns an iterable of (member, scalar) tuples
# def with_matched_version_of(self, other:OtherType) -> Iterable[Tuple[MemberType, OtherType]]: ...
# @overload  # with_matched_version_of(self, matching_sequence) returns an iterable of (member, othermember) tuples
# def with_matched_version_of(self, other:Sequence[OtherType]) -> Iterable[Tuple[MemberType,OtherType]]: ...
# def with_matched_version_of(self, other):
#     """Returns an iterator of (m, o) pairs, where each m will be a successive member of self (so if self has
#        just 1 member, the iterator will return just 1 pair), and each o will be based on other, and will
#        either be a corresponding part of other (if other is a non-string with the same length as self),
#        or will broadcast other's single element repeatedly (if other is a length-1 non-string sequence),
#        or else will broadcast other itself repeatedly.
#        Raises ValueError if other is not broadcastable with self"""
#     # TODO could generalize this to multiple args, as with broadcast_together.  Could even merge with keyword arg.
#     # TODO take care with the case where other is a mere iterator; this currently broadcasts it as is!  Desired?
#     try:
#         n = len(self)
#     except:  # wrap scalar self as 1-tuple
#         self = (self,)
#         n = 1
#     if n == 0:
#         return iter(())
#     return zip(self, broadcast_to_length(other, n))

# @overload  # broadcast_together(self, scalar) returns an iterable of (member, scalar) tuples
# def broadcast_together(self, other:OtherType) -> Iterable[Tuple[MemberType, OtherType]]: ...
# @overload  # broadcast_together(self, matching_sequence) returns an iterable of (member, othermember) tuples
# def broadcast_together(self, other:Sequence[OtherType]) -> Iterable[Tuple[MemberType,OtherType]]: ...
def broadcast_together(*sources, match_first = False):
    """Returns an iterator of tuples (m0, m1, ...), where each m will be drawn from the corresponding source,
       broadcasting each source as needed.  If match_first, then other sources will be broadcast to match the
       length of the first.  Otherwise, if all sources with length other than 1 have the same length, then
       scalar sources and the singular contents of length-1 sources will be broadcast to that length.
       If sources disagree on which length to use, a ValueError is raised."""

    # TODO generalize this to have dictionaries match keys with each other rather than vectorizing iterationwise
    #   0. If no dictionaries, just zip sequences together
    #   1. If mix of dictionaries and non, use an enumeration range to index the dict(s)
    #   2. If just dictionaries, use first one's keys to index the dicts.

    # TODO this is a bit tricky if we want to accommodate both (a) containers that are accessable by iteration only
    #  like sets and dictionary views, and (b) containers that we want to access by index/key only like dicts.
    #  Also, even for those sequence-like containers that allow indexed access, we might rather access these by
    #  iterating.

    # TODO another bit of trickiness would arise if we'll allow super-dictionary operands and/or super-list operands.
    #  Could go back to a policy of saying the first non-broadcastable operand sets the standard that everyone else
    #  must comply with.  That means sub+super will be fine, but super+sub would cause an error.

    # TODO our goal is probably to (1) determine what indices to use, including figuring out n, the number of members
    #  the output will have, (2) create an iterator for each operand that will yield
    #  its relevant values, or raise an error if it can't, (3) zip these iterators together

    if not sources: return ()
    sources = tuple(as_sized_iterable(s) for s in sources)

    # Identify the model whose length n we will broadcast singletons to
    if match_first:
        model = sources[0]
    else:
        # The model will be the first source that can't be broadcast, either due to non-1 size or to being a Mapping
        model = next((s for s in sources if len(s) != 1 or isinstance(s, Mapping)), sources[0])
    n = len(model)

    # If the model and all non-broadcastable sources are Mappings we'll use the model's keys as our indices
    if isinstance(model,Mapping) and all(len(s)==1 or isinstance(s, Mapping) for s in sources):
        indices = model.keys()
    else: # otherwise we'll use integer indices ranging 0...(n-1) and expect any involved mappings to have these as keys
        indices = range(n)

    return zip(*(broadcast_to_indices(s, n, indices) for s in sources))

    # lengths = set(len(s) for s in sources)  # Should be {1}, {n}, or {1, n}
    # if match_first:
    #     n = len(sources[0])
    #     if not lengths <= {1, n}:  # If the set contains votes for any other broadcast length, raise an error
    #         raise ValueError(f"Unable to broadcast operands with lengths {lengths} to match length {n}.")
    # else:
    #     if len(lengths)>1: lengths -= {1}   # If there's a higher priority candidate n, remove the votes for 1
    #     if len(lengths)>1:                  # But if the set still disagrees about broadcast length, raise an error
    #         raise ValueError(f"Unable to broadcast operands with lengths {lengths}")
    #     n = lengths.pop()                   # Retrieve the one remaining length
    # if n == 0: return iter(())              # If any source is length-0, we can simply yield no tuples.
    # return zip(*(s if len(s) != 1 else itertools.repeat(next(iter(s)), n) for s in sources))

def with_matched_version_of(*sources):
    """Returns an iterator of tuples (m0, m1, ...), where each m will be drawn from the corresponding source,
       broadcasting each source as needed to match the first, so if the first has just 1 element, then all must.
       If any sources disagree with the first about a non-1 length to use, a ValueError is raised."""
    return broadcast_together(*sources, match_first=True)

# TODO this will likely eventually be replaced by BroadcastHandler
def broadcast_args(fn, args, kwargs, match_first = False):
    """Returns an iterator of (f, a, k) triples, where each f will be a drawn from fn (typically an EachFunction or
       EachContainer of bound methods), each a will be a tuple drawn from args, and each k will be a dictionary
       based on kwargs.  All of these will be broadcast as appropriate, to match the size of fn if match_first is true,
       or to match their agreed-upon length other than 1 otherwise.  If sources disagree upon their preferred length,
       a ValueError will be raised."""
    # We will string these together, pass them into broadcast_together, then package the results it yields into a triple
    start_k = 1 + len(args) + 1  # fn takes 1, args take len(args), kwargs take the rest
    return ((f_a_k[0], f_a_k[1:start_k], {key:value for key,value in zip(kwargs.keys(), f_a_k[start_k:])})
            for f_a_k in broadcast_together(fn, *args, *(kwargs.values()), match_first=match_first))


class BroadcastHandler:
    """`B = BroadcastHandler(*sources, match_first = False)` creates a simple object that can be used to handle
        broadcasting the sources together for a vectorized operation.  sources[0] is presumed to be an EachContainer.
        This creates a sized iterable version of each source (spieling iterables into tuples if needed), introspects
        the sources to determine B.n the length of their broadcast version and B.indices the keys/indices that will
        be matched across distributed sources, either range(n) when sequence-like sources are involved, or the
        keys of an involved dictionary.
        `for tup in B` will yield successive n-tuples of values drawn from each source, broadcasting as needed.
        `B(it)` will return a new EachContainer containing the output of such a broadcast operation, where it is an
        iterator of successive values to be stored in this output container.  If B.indices is a range, this output will
        be an EachContainer of sources[0]._each_output_type.  Otherwise it will be an EachMapping mapping each
        of B.indices to the corresponding value.
        This allows overloaded functions like EachContainer.__add__ to simply create B, and define a generator to
        iterate through broadcast values from the sources and yield the resulting value (in this case their sum), and
        let B do the rest, e.g. with `return B(s + o for s, o in B)`.  Having __add__ define its own iterator allows
        the inner loop of a vectorized operation to occur without any function calls, for efficiency."""
    def __init__(B, *sources, match_first = False):
        # TODO consider allowing iterable sources, construing them as non-broadcastable (tricky if one ends up being model)
        sources = tuple(as_sized_iterable(s) for s in sources)
        B.sources = sources
        B.E = sources[0]  # type: EachContainer

        # Identify the model whose length n we will broadcast singletons to
        if match_first:
            B.model = sources[0]
        else:
            # The model will be the first source that can't be broadcast, either due to non-1 size or to being a Mapping
            B.model = next((s for s in sources if len(s) != 1 or isinstance(s, Mapping)), sources[0])
        B.n = len(B.model)

        # If the model and all non-broadcastable sources are Mappings we'll use the model's keys as our indices
        if isinstance(B.model, Mapping) and all(len(s) == 1 or isinstance(s, Mapping) for s in sources):
            B.indices = B.model.keys()
        else:  # otherwise we'll use integer indices ranging 0...(n-1) and expect any involved mappings to have these as keys
            B.indices = range(B.n)

    # TODO figure out how to @overload these to properly type-hint the EachMapping possibility
    def __call__(B, it:Iterable[Callable[[Any], OutputType]]) -> 'EachContainer[OutputType]':
        # TODO think about whether I need to do anything to handle nesting.
        #  Suppose C contains vectorlike lists, and we've defined E = each(C, nested=2).
        #  Now suppose someone did E+1.  As this was computed, iterating over E should have recognized that each
        #  of the member vectors itself needed to be eachified and taken care of passing any cues for further
        #  eachification down the chain, so the content should now have been appropriately eachified, and then
        #  that eachification should persist through whatever operation was performed on them, so the outermost
        #  layer of the new output can assume that its contents are eachified enough, so can have nested = 1
        if isinstance(B.indices, range):
            return B.E._each_output_type(list(it), nested=1)
        return EachMapping({key: value for key, value in zip(B.indices, it)}, nested=1)

    # TODO figure out how to @overload these to properly type-hint the EachMapping possibility
    def in_place(B, it:Iterable[MemberType]) -> 'EachContainer[MemberType]':
        E = B.E
        if hasattr(type(E.whole), '__setitem__'):    # If E.whole's items are settable, we'll alter each in place
            for index, value in zip(B.indices, it):
                E.whole[index] = value
        else:                                        # Otherwise we'll replace E.whole with new content
            E.whole = E._each_output_type(it)
        return E  # in place operations require that the first operand be returned

    def __iter__(B):
        return zip(*(broadcast_to_indices(s, B.n, B.indices) for s in B.sources))


def repeat_if_singular(source)->Iterable:
    """If source is a string or non-iterable scalar, it will be repeatedly yielded, indefinitely.
       If source is an iterable that yields just one value, that value will be repeatedly yielded, indefinitely.
       Otherwise, this is equivalent to iterating source."""
    # If source is a string or non-iterable scalar, we'll simply broadcast/repeat it
    if not isinstance(source, Iterable) or isinstance(source, str):
        while True: yield source
    source = iter(source)
    try:
        first_value = next(source)
    except StopIteration:  # if there is no first value, there's nothing to repeat!
        return
    yield first_value
    try:
        second_value = next(source)
    except StopIteration:  # If there is no second value, then we'll simply broadcast/repeat the first_value
        while True: yield first_value
    yield second_value  # If there are at least two values, we'll just yield all the values from the source
    yield from source

# def with_matched_args(self, args, kwargs) -> Iterable[Tuple[MemberType, Tuple, dict[str, any]]]:
#     """Returns an iterator of (m, a, k) triples, where each m will be a successive member of self,
#        each a will be a tuple drawn from args, broadcasted where necessary to match self,
#        and each k will be a dictionary based on kwargs, again broadcast, if necessary, to match self.
#        This iterator can provide relevant args and kwargs for successive calls of a distributed method."""
#
#     # args_it will yield successive tuples of args, ready to send as *args to successive calls of some function
#     if args:
#         args_it = zip(*(self.matched_version_of(a) for a in args))
#     else:                                                     # or if no args were given, then just
#         args_it = itertools.repeat(args, len(self.whole))  # repeatedly yield that empty tuple
#
#     if kwargs:
#         # kw_values_it will yield similar tuples of values to associate with the **kwargs in that call
#         kw_values_it = zip(*(self.matched_version_of(value) for value in kwargs.values()))
#         # kw_dicts_it will yield dictionaries by zipping each key together with its value from that tuple
#         kw_dicts_it = (dict(zip(kwargs, kw_values)) for kw_values in kw_values_it)
#     else:                                                          # or if no kwargs were given, then just
#         kw_dicts_it = itertools.repeat(kwargs, len(self.whole)) # repeatedly yield that empty dict
#
#     # returned iterator will yeild (m, a, k) tuples: m = next member of self, a = tuple of args, k = dict of kwargs
#     return zip(self, args_it, kw_dicts_it)

# === each factory ===

#TODO add kwargs to each overloading?
@overload
def each(string: str) -> 'EachContainer[str]': ...
@overload
def each(s: Set[MemberType]) -> 'EachSet[MemberType]': ...
@overload
def each(it: Iterable[MemberType]) -> 'EachContainer[MemberType]': ...
@overload
def each(m: MutableMapping[KeyType, MemberType]) -> 'EachMapping[KeyType, MemberType]': ...
@overload
def each(m1:MemberType, m2:MemberType, *members: MemberType) -> 'EachContainer[MemberType]': ...
# Another possibility is each(m1:MemberType, enlist=False) -> 'MemberType'  # left undeclared to not confuse linters
def each(*members, nested: Union[int, bool, None] = next, enlist = True) \
        -> Union['EachContainer', 'EachMapping', 'EachSet']:
    """`each(object)` returns an "each-wrapped object" which for many purposes behaves like the wrapped object itself,
       except that most operations involving each-wrapped objects will be distributed over containers' members,
       with singular operands being "broadcast" or repeatedly used.
       **EachContainers**:
       `each(collection)` returns an each-wrapped version of the given (iterable and sized) collection.
       `each(iterator)` spiels an unsized iterable (like a generator expression) into each(list(iterator)) if `enlist`
       is its default value True; TODO otherwise NotImplemented
       `each(member1, member2, ...)` creates an each-wrapped list of the given members.
       `each(member1)` for scalar member1 is equivalent to `each([member1])` if `enlist` is its default value True;
        otherwise this will return member1 as is.
       `each(1, 2, 3) * 5` multiplies each member by 5, yielding each(5, 10, 15). (I.e. it "broadcasts" the 5.)
       `each(1, 2, 3) + (2, 3, 4)` does vectorized or pairwise addition, yielding each(3, 5, 7).
       `each(motors).velocity` returns the .velocity of each motor (in an each-wrapped list)
       `each(motors).velocity = 0` sets the .velocity of each motor to 0. (i.e. it broadcasts the 0).
       `each(motors).velocity = (1, -1)` sets the .velocity of the first to 1, the second to -1, if there are two.
       `each(motors).velocity *= 0.5` halves the .velocity of each motor.
       `each(C).method(arg1, arg2, ...)` calls m.method(a1, a2, ...) for each m in C, with args broadcast as needed.
       `math.sqrt(each(1,4,9))` would raise an error because sqrt doesn't know how to handle a list of inputs.
       `each(math.sqrt)((1,4,9))` returns each square root, i.e. each(1,2,3). I.e. wrapping math.sqrt in each() creates
       a version of sqrt that *does* know how to perform distributed operations with broadcasting.
       `each(container)[spec1, spec2, ...]` uses the first given spec to index/slice the container, and then any
       remaining specs to index/slice each of the members.  If any spec is itself each-wrapped, it is distributed.
       `E > 5` will return an each-wrapped list of booleans indicating which members of E are more than 5.
       `E[E > 5]` returns an EachContainer containing those members of E that are greater than 5.
       `E[list_of_indices]` returns `each(E[i] for i in list_of_indices)`, i.e. it looks up each given index in E
       **EachMappings**:
       `ED = each(D)` creates an each-wrapped version of dictionary D.
       `each(D)+1` produces a new EachMapping like D, except each key is mapped to a value that is 1 greater.
       `each(D1) + D2` produces a new EachMapping that maps each key k from D1 to D1[k]+D2[k].  Note that D2 will need
       to accept the same keys/indices as D1. (D2 could be another dictionary, or a sequence if D1's keys are ints).
       `each(D).key` is a set-like EachContainer of D's keys.
       `each(D).keys()` is a way of accessing D.keys(), the Python dictionary-view of D's keys.
       `each(D).values()` is a sequence-like container of D's values (each-ified if the container is nested).
       `ED[key]` returns `D[key]` like an ordinary dictionary, though key must be non-iterable or encapsulated in [].
       `ED[key]` = v` alters the value for key in D like an ordinary dictionary, but iterable keys again need [].
       `ED[[key1, key2, ...]]` returns TODO could plausibly return subdict, or values
       `ED[[key1, key2, ...]] = v` alters values in D associated with the given keys, broadcasting v to match given keys
       `ED[ED.value > 5]` returns TODO again could plausibly return subdict or its values
       `ED[ED.value>5] = v` alters the values in D for which the condition is True, broadcasting singleton v
        or looking up these keys/indices in container v
       `ED[domain, range]` = v` peels off the domain index to select items in ED (using any of the above formats) and
          then alters each of those values by setting `value[range] = v`
       **EachSets**
       TODO
       **Nesting**
       The optional `nested` parameter affects whether operations will also be distributed over submembers of members of
       this EachContainer.  Suppose `rows` is a list of rows, each of which is itself a list of cells.
       `each(rows)` will distribute operations to each row, but not further within each row.
       `each(C, nested=next)`, which is the default, will add one extra layer of each-distribution beyond what C has.
       `each(each(rows))` therefore will distribute ops over each cell in each row, akin to a nested for loop.
       `each(rows, nested=2)` will distribute ops over two levels (each cell of each row) but not further.
       `each(rows, nested=True)` will distribute ops all the way down, good for traversing trees of arbitrary depth.
       `each(C, nested=None)` and `each(C, nested=1)` treat C eachwise, but do not add further eaching, even if C
       is already an EachContainer (unlike the default `nested=next` which would add a layer of eaching).
       `each(C).each.attr` is short for `each(each(C).attr, nested=1)` or for `each(each(m.attr, nested=1) for m in C`.
       In all these cases, if submembers like rows' cells were already EachContainers then ops on them will be
       distributed for that reason, even if a higher-level `nested` setting was not causing them to be distributed.
       **Notes**
       Note1: Vectorized each-operations generally rely upon having containers that match in keys/indices.
       Beware mixing containers created before and after elements have been removed/added to some list or dictionary!
       In binary operations and multi-argument function calls the indices of the first non-broadcast sequence will be
       used, or if all non-broadcast operands are mappings, then the first one's keys will be used.  It can be
       acceptable to have other operands be larger with additional keys/indices.
       Note2: Even though Python strings are technically "sequences", each-wrapped objects generally treat them as being
       atomic and won't distribute them into constituent letters in cases where other sequences would be distributed.
       `each(flags).color = 'red'` sets each flag.color to 'red', not the first to 'r', the second to 'e', and so on...
       Note3: In each-broadcasting, lengthless scalars (like 1 and 'red') and length-1 sequences (like [1] or [(1,2,3)])
       will be "broadcast" or repeatedly used, whereas operations will be distributed over other iterable operands.
       This is much like NumPy's broadcasting, except EachContainers begin matching at the *outermost* dimension of
       nested EachContainers, whereas Numpy begins matching at the *innermost* dimension of multi-dimensional arrays.
       (NumPy arrays are perfectly rectangular, so it is straightforward to find their innermost dimension, whereas
       nested EachContainers needn't be uniform in length or even depth so needn't have a unique "innermost dimension".)
       You may encapsulate an operand in [] to delay when it will be distributed in ops involving nested EachContainers.
       `each('A','B') + each('1','2')` distributes both immediately, returning each('A1', 'B2'). In contrast,
       `each('A', 'B') + [each('1', '2')]` distributes the letters on the outermost dimension, but encapsulates the
       numbers so they are distributed on an inner dimension, returning `each(each('A1','A2'), each('B1','B2'))`.
       (This serves a similar purpose to using numpy.reshape to alter which dimension an array will broadcast at.)"""
    if nested is True or nested is all: nested = float('inf')
    if nested is None or nested is False: nested = 1
    if len(members) == 1 and not isinstance(members[0], str):
        source = members[0]
        if isinstance(source, Iterable):
            if isinstance(source, EachContainer):
                if nested is not next: nested = max(nested, source._each_nested)
                source = source.whole
            else:
                if nested is next: nested = 1  # Source itself is the "next"/only non-EachContainer to be eachified
            if isinstance(source, Set):
                return EachSet(source, nested=nested)
            if isinstance(source, Mapping):
                return EachMapping(source, nested=nested)
            if not isinstance(source, Sized):
                if enlist:
                    return EachContainer(list(source), nested=nested)
                else:
                    raise NotImplementedError  # TODO allow a version of EachIterable
            return EachContainer(source, nested=nested)
        if not enlist:
            return source  # if source is non-eachable scalar and we aren't supposed to enlist it, return it as is
    if nested is next: nested = 1  # This will be the "next"/only non-EachContainer to be eachified
    return EachContainer(list(members), nested=nested)


# === EachAttributeGetter ===

class EachAttributeGetter:
    """`each(X).each.attr` returns a nested structure of EachContainers, equivalent to `each(each(x.attr) for x in X)`.
       An EachAttributeGetter is returned by `each(X).each` and mediates getting each versions of attributes."""

    # TODO NotImplemented

    # TODO consider what `each(each(X))` should return?
    #  One plausible rule says that each(foo) always returns a container much like foo, except many operations will be
    #  distributed over its top-level members.  Since each(X) is already such a container, `each(each(X))` wouldn't
    #  change its overall functionality, though it might make this functionality *slower* if each additional layer takes
    #  its own crack at distributing operations, so on this approach it might make more sense to make each(E)
    #  return E itself when E is already an EachContainer.  Intuitively each(foo) means "foo construed eachwise"
    #  so if foo was already being construed eachwise, adding an extra each serves only as insurance.
    #  A second plausible rule says that each(foo) always returns a container that *further* distributes operations
    #  a level deeper than they would already have been distributed by foo.  So each(table) would distribute operations
    #  over rows of the table, and each(each(table)) would further distribute operations over each cell of each row,
    #  I.e. it would mean "each cell in each row of table".  This helps to make each instance of 'each' serve as an
    #  additional 'for'-like iterator. This approach might run into some annoyances if each is applied to something
    #  that is already an EachContainer, leading to a common code structure being
    #  `(C is isinstance(C, EachContainer) else each(C))` or perhaps a helper function like `eachwise(foo)`).
    #  This can also run into annoyances with turning nested scalars into little each-containers.
    #  A third plausible rule is a slight twist on the second, saying that each(each(foo)) will distribute operations
    #  over an additional layer of non-string iterables within foo, but won't just blindly each-ify scalars in foo.
    #  This would avoid some cases of the latter annoyance with the second rule.
    #  There might also be some use for each_nested(foo) that returns each_nested versions of each iterable within foo,
    #  so e.g. if you have a tree that might have arbitrarily deep layers of nesting, each_nested(tree) would
    #  reach each of the leaves, no matter how deep they are, preserving the full tree structure in each output.
    #  A fourth somewhat-plausible rule would be to always read 'each' as meaning 'each_nested' though there are
    #  likely many cases where you want to do something with each member of an iterable without automatically delving
    #  into nested iterables!




# === EachContainer ===

class EachContainer(Generic[MemberType]):
    """TODO update this to match the `each` docstring which is more uptodate
       TODO figure out whether each() always converts contents to list, only turns iterators/generators to lists,
        or always leaves its content as-is.  There is something attractive about leaving everything as is, making
        each always be a light-weight wrapper (though its outputs won't be!).  However, this would prevent some
        broadcast calculations for iterators and could lead to surprising cases where each(i for i in C if f(i)) can
        only be used once, requiring cluttering it, e.g., as each([i for i in C if f(i)])
        each(huge_iterator) would, by default, be able to do exactly one thing to each member, though potentially could
        do more with appropriate use of itertools.tee, though providing an interface for that would be tricky,
        especially since Python will try to incrementally compute the full depth of complex expressions, so e.g.,
        even if we could indicate the number of intended uses, e.g. with E = each(huge_iterator, uses=2), an
        expression like `E.x**2 + E.y**2` would still go through a branch of the whole iterator on the x-side of the tee,
        before even starting on the y-side of the tee, making this equivalent to forming a list anyway.  Given that
        limitation in Python, it seems like each(huge_iterator) can have iterator-like efficiency only in very simple
        single-use and directly-linked double-use cases like E.x * E.y, so probably better not to invest any complexity
        in wrapping iterators, and instead have `each` convert these to something that'll behave more like other wrappees.

       TODO will need to take care with modify commands about modifying a container while iterating it...

       TODO it might be convenient for EachContainers to have a .each method to make it more natural to do nesting.
        `each(segments).each.wheels`  is so much more intuitively clear than `each(each(segments).wheels)`!!!
        `each(segments).each.wheels = each(each(1,2))`
        `each(segments).each.wheels = each.each(1,2)`  here the bare `each` works as a capsule"""

    # TODO could consider re-using the more complex output-type selection from Vectors?
    _each_output_type: type        # The type of output ops will produce (will be set once this class is defined)
    whole: Collection[MemberType]  # Will store the members of this EachContainer

    # TODO consider what 'for item in each(C)` should yield. One plausible reading is that this just means `for i in C`,
    #  so in this case the `each()` was redundant with the `for` in that both said we would treat C eachwise.
    #  But another plausible reading is that `each` and `for` each signify their own layer of iteration. So
    #  `for item in C` means `for any item you find by going through C itself` and
    #  `for item in each(C)` means `for any item you can find by going through each member of c in succession`
    #  This latter reading makes `for leaf in each(tree, nested=True)` traverse the entire tree, which seems good.
    #  I'll need to take care to make this interact right with other iteration machinery, like broadcasting and getitem,
    #  though these can probably be made to iterate over .wholes or perhaps .values() and be fine.

    # TODO consider what 'for m in NE' should do when NE is a nested each-container.  It is somewhat plausible
    #  to think that `for cell in each(each(rows))`

    # TODO consider whether this should be rewritten as __new__ so EachContainer(scalar) could return scalar itself
    #  This would help simplify cases where a would-be nested supercontainer wraps each of its members in EachContainer
    #  But it may be that we want something more like each() for this, which can flexibly choose a container type
    #  Though we have each(scalar) return each([scalar]) which would be incompatible with having it return scalar.
    #  Probably best to work through some examples.
    #  `for m in each(C)` should return members of C as they are, since C was not yet an each-container.
    #   This case would probably be handled by converting nested to 1 in container creation when given whole is non-each.
    #  `for m in each(E)` is supposed to turn each-members of E into versions that retain nested=next,
    #   turn other eachable members into each(m, nested=1), and leave scalar members alone.
    #  each(E, nested=n) leave scalar members alone, and turn everything else into an each with nested at least n-1.
    #  each(E, nested=1) is supposed to leave all members alone, which would be special case of the above if:
    #  each(E

    def __init__(self, whole: Iterable[MemberType], nested: Union[int, bool, None] = 1):
        if not isinstance(whole, Sized): whole = list(whole)
        self.__dict__['whole'] = whole
        self.__dict__['_each_nested'] = nested

    def __repr__(self):
        # In the common case of each([1, 2, ...]) we abbreviate to the equivalent each(1, 2, ...)
        if isinstance(self.whole, list):
            if len(self.whole) != 1 or not isinstance(self.whole[0], Iterable):
                return f"each({repr(self.whole)[1:-1]})"
        # If a list has a 1 iterable member, or if we're each-wrapping a non-list, abbreviation might not be equivalent
        return f"each({repr(self.whole)})"
        # TODO consider making a better __repr__ for EachContainers of bound methods like {parent}.{method.__name__}
        #  This would probably require caching the parent or its __repr__ on object creation

    # Python requires that bool return True or False, so no option to vectorize.  For that, use `each(bool)(E)`
    def __bool__(self):
        return bool(self.whole)

    # TODO similarly for __int__,  __float__?

    # --- each container interface ---

    # TODO decide whether to live with the incongruence between having len(E) be a list of number of members and
    #      having iter(E) iterate through descendants.
    def __len__(self) -> int:
        return len(self.whole)

    # TODO could have overloaded type signature
    @property
    def themselves(self) -> Iterable[MemberType]:
        """Yields members of this EachContainer, wrapping them in each() if this EachContainer has provisions for
           nesting.  If this EachContainer does not have any special nesting instructions, this will be equivalent
           to iter(self.whole) for non-Mappings, and to iter(self.whole.values()) for for Mappings.
           If this container does have nesting instructions, e.g. nested = next, or nested > 1, then this will wrap
           the contained items in each(), propagating nesting instructions downwards appropriately."""
        # TODO consider whether/how to merge this with .values()
        # TODO consider whether I want this to be more durable and sized than a mere iterable
        if self._each_nested is next or self._each_nested > 1:
            new_nested = next if self._each_nested is next else self._each_nested - 1
            return (each(i, nested = new_nested, enlist=False) for i in self.whole)
        return iter(self.whole)

    @property
    def key(self) -> 'EachContainer[int]':
        """Returns an EachContainer of keys/indices used in this EachContainer.  For sequence-like EachContainers,
           this will be each(range(n)), where n is the length of the container, i.e. the integers 0 ... n-1.
           This is provided for parity with EachMappings, but can serve many of the same purposes as enumerate()."""
        return EachContainer(range(len(self.whole)))

    @property
    def keys(self) -> 'Sequence[int]':
        """Returns a sequence of keys/indices used in this EachContainer.  For sequence-like EachContainers,
           this will be range(n), where n is the length of the container, i.e. the integers 0 ... n-1."""
        return range(len(self.whole))

    def values(self) -> Iterable[MemberType]:
        """This is a generalization of dict.values() to other EachContainers, iterating through the indexed values
           in the top level of the container.  If this container does not have any special nesting instructions,
           this will be equivalent to iterating self.whole for non-Mappings, and to iterating self.whole.values()
           for Mappings.  If this container does have nesting instructions, e.g. nested = next, or nested>1,
           then this will wrap the contained items in EachContainers, propagating nesting instructions downwards."""
        # TODO consider whether I want this to be more durable and sized than a mere iterable
        if self._each_nested is next or self._each_nested > 1:
            new_nested = next if self._each_nested is next else self._each_nested - 1
            return (each(i, nested=new_nested, enlist=False) for i in self.whole)
        return iter(self.whole)

    def __iter__(self) -> Iterable[MemberType]:
        """`for i in each(C)` yields each item in C. Nested EachContainers are distributed into separate items."""
        for i in self.values():
            if isinstance(i, EachContainer):
                yield from i
            else:
                yield i

    def __reversed__(self) -> Iterable[MemberType]:
        return iter(reversed(self.whole))

    # Python always converts `m in C` checks to simple booleans, so we can't really distribute __contains__ itself
    def __contains__(self, item):
        return item in self.whole

    def contains(self, *items) -> 'EachContainer[bool]':
        """A distributable version of Python's __contains__, useful for testing whether another item or items is in
           each member of this EachContainer, with this EachContainer or the given item broadcast to match the other.
           Returns an EachContainer of boolean True/False values, indicating the results of each containment test.
           The items may be given separately or as a single non-string iterable.  If you want to test whether
           a single non-string iterable itself is contained, encapsulate that candidate in [] to broadcast it as is.
           `each('abc', 'xyz').contains('a', 'b')` returns each(True, False) since 'a' in 'abc', but 'b' not in 'xyz'
           `each('abc', 'xyz').contains('a')` returns each(True, False) since 'abc' contains 'a' and 'xyz' doesn't.
           `each('abc').contains('a', 'x')` returns each(True, False) since 'a' is in 'abc' but 'x' isn't.
           `each([area]).contains(targets)` broadcasts the area TODO ...
           Note that in the second example 'a' is broadcast for use in each comparison, as is 'abc' in the third.
           `E1.contains(E2)` and `E2.is_in(E1)` are equivalent, and will be used to handle nested eachContainers."""
        # If given a single target iterable, unpack it so BroadcastHandler won't see it as encapsulated
        if len(items)==1 and isinstance(items[0],Iterable) and not isinstance(items[0], str):
            items = items[0]
        B = BroadcastHandler(self, items)
        return B(s.contains(i) if isinstance(s, EachContainer) else
                 i.is_in(s)    if isinstance(i, EachContainer) else
                 i in s
                 for s, i in B)

    def is_in(self, *containers) -> 'EachContainer[bool]':
        """A distributable way of testing whether each member of this container is a member of another container or
           containers, broadcasting this EachContainer or the other container to match the other.
           Returns an EachContainer of boolean True/False values, indicating the results of each containment test.
           The other container(s) may be given separately or as a single non-string iterable.
           `each('a','b').is_in('abc', 'xyz') returns each(True, False) since 'a' is in 'abc', but 'b' is not in 'xyz'.
           `each('a','x').is_in('abc') returns each(True, False) since 'a' is in 'abc' and 'x' isn't.
           `each('a').is_in('abc','xyz') returns each(True, False) since 'a' is in 'abc' but not in 'xyz'.
           Note that in the second example, 'abc' is broadcast for use in each comparison, as is 'a' in the third.
           `E1.contains(E2)` and `E2.is_in(E1)` are equivalent, and will be used to handle nested EachContainers."""
        if len(containers)==1 and isinstance(containers[0],Iterable) and not isinstance(containers[0], str):
            containers = containers[0]
        B = BroadcastHandler(self, containers)
        return B(s.is_in(c)    if isinstance(s, EachContainer) else
                 c.contains(s) if isinstance(c, EachContainer) else
                 s in c
                 for s, c in B)

    # E[...] can return various types depending on what indices are given, so type-hinting requires @overload
    @overload  # E[index] returns a single member
    def __getitem__(self, index:int) -> MemberType: ...
    @overload  # E[slice], E[boolean_mask] or E[sequence_of_indices] returns an EachContainer of relevant members
    def __getitem__(self, item:PluralIndexType) -> 'EachContainer[MemberType]': ...
    @overload  # E[index1, index2] returns whatever m[index2] returns
    def __getitem__(self:'EachContainer[MemberType[SubmemberType]]', item:Tuple[int, int]) -> SubmemberType: ...
    @overload  # E[index, slice] returns whatever m[slice] returns, presumed to be another container like m
    def __getitem__(self, item:Tuple[int, slice]) -> 'MemberType': ...
    @overload  # E[slice, index] returns an EachContainer of whatever sort of submember m[index] returns
    def __getitem__(self:'EachContainer[MemberType[SubmemberType]]',
                    item:Tuple[PluralIndexType, int]) -> 'EachContainer[SubmemberType]': ...
    @overload  # E[slice1, slice2] returns an each-container of whatever m[slice2] returns, presumed to be like m
    def __getitem__(self:T, item:Tuple[PluralIndexType, PluralIndexType]) -> T: ...
    def __getitem__(self, item):
        """`E[i]` returns the value in EachContainer E with index/key i, for non-iterable or string i.
           `E[start:stop:step]` or other slices like `E[:stop]` or `E[start:]` return each item in E satisfying the
             slice. If E.whole does not support slicing, this will be approximated, returning a subcontainer whose
             indices in E satisfied start <= i < stop, though for mappings step will be ignored, and for other
             iterables negative start and stop will be converted to positive, and negative step will raise an error.
             If E is a mapping, this returns a submapping with the same keys. Otherwise it returns a subsequence
             with its indices automatically renumbered (since sequence indices always start at 0 and count up).
           `each[D]['a':'n']` therefore returns the subdict of dict D whose keys start with 'a' through 'm'.
           `E[boolean_mask]`, where boolean_mask is an iterable whose values are all True/False, returns each item of E
             such that boolean_mask[i] is True (treating non-mapping iterables as 0-indexed sequences, as always).
             If E is a mapping the result is a submapping; otherwise it is an automatically renumbered subseqeunce.
           `E[E > 0]` therefore returns the submapping/subsequence of E whose values are positive.
           `E[old_indices]` for non-tuple sequence/mapping old_indices returns a new container F,
             with F[new_index] == E[old_indices[new_index]] treating iterables as 0-indexed sequences, as always.
             Note: if old_indices is a tuple, then Python automatically construes it as the E[domain, range] case below!
           `E[[1,3,5]]` is therefore equivalent to `each(E[1], E[3], E[5])` with implicit new indices [0, 1, 2].
           `each({'a':1, 'b':2})[{'A':'a'}]` is therefore equivalent to each('A':1), with the explicit new index 'A'
              mapping to the same value as did the corresponding old index 'a' (namely 1).
           `E[domain, range]` peels off the first index `domain` to select members of E in any of the above ways, and
              then uses any remaining indices to index within each of those members, broadcasting as appropriate.
           `E[:, range]` therefore returns each m[range] for each member m in E[:], which is the entirety of E."""

        # -- Peel off the domain_index (indexing this EachContainer) from range_indices (indexing in domain members) --
        if isinstance(item, tuple) and len(item)>=1:
            domain_index = item[0]    # peel off the first index to slice the EachContainer itself
            range_indices = item[1:]  # remaining index/indices will slice each member
        else:
            domain_index = item
            range_indices = ()

        output_type = EachContainer  # default setting; may overwrite to self._each_output_type or to EachMapping
        domain_is_singular = False   # default setting; will be overwritten when extracting single member
        domain: Union[MemberType, Iterable[MemberType], Mapping[KeyType, MemberType]]

        # -- Calculate the domain of relevant member(s) within this EachContainer --
        if domain_index == slice(None, None, None):  # Processing E[:], full slice, so no need to index/slice E
            if not range_indices: return self        # E[:] just is E itself
            domain = self.whole                      # E[:, range] uses E.whole as domain, indexes into each member
            output_type = self._each_output_type
        elif isinstance(domain_index, slice):        # When the domain is a partial slice...
            try:                                     # First we see if self.whole knows how to handle slices
                domain = self.whole[domain_index]
            except:                                  # But if it doesn't, we then must approximate this ourselves.
                start, stop, step = domain_index.start, domain_index.stop, domain_index.step
                if isinstance(self, EachMapping):
                    domain = {key:value for key, value in self.items()
                                        if (start is None or start <= key) and (stop is None or key < stop)}
                    output_type = type(self)  # EachMapping or EachSet
                # # TODO This seems intuitively right for sets, but haven't yet committed to a way of handling these
                # elif isinstance(self.whole, set):
                #     domain = {m for m in self.whole if (start is None or start <= m) and (stop is None or m < stop)}
                else:
                    if start < 0: start += len(self.whole)
                    if stop  < 0: start += len(self.whole)
                    domain = itertools.islice(self.whole, start, stop, step)
        elif isinstance(domain_index, Mapping):  # Processing E[mapping]
            # A boolean mask like E[E>1] would map keys/indices to True/False values
            if all(x is True or x is False for x in domain_index.values()):
                if isinstance(self, Mapping):  # A boolean mask on a mapping returns a submapping
                    domain = {key:value for key,value in self.items() if domain_index[key] is True}
                    output_type = type(self)  # EachMapping or EachSet
                else:                                # A boolean mask on a sequence returns a subsequence
                    domain = (value for index, value in enumerate(self.whole) if domain_index[index] is True)
            else: # Processing E[mapping] where the mapping maps new_indices to old_indices
                # if self.whole doesn't support getitem (e.g., because it is a set or dictview), spiel it into a tuple
                whole = self.whole if hasattr(type(self.whole), '__getitem__') else tuple(self.whole)
                domain = {new_index: whole[old_index] for new_index, old_index in domain_index.items()}
        elif isinstance(domain_index, Iterable) and not isinstance(domain_index, str):  # Processing E[iterable]
            try:  # Check if the given domain_index has introspectable length
                length = len(domain_index)
            except TypeError:  # If it doesn't (e.g. due to being a raw generator) we'll spiel it into a tuple
                domain_index = tuple(domain_index)
            # A boolean mask like E[E>1] would be a length-matched sequence of True/False values
            if len(domain_index) == len(self.whole) and all(x is True or x is False for x in domain_index):
                if isinstance(self.whole, Mapping):
                    whole = self.whole
                    domain = {index: whole[index] for index, boolean in enumerate(domain_index) if boolean is True}
                    output_type = EachMapping
                else:
                    domain = (value for value, boolean in zip(self.whole, domain_index) if boolean is True)
            else:  # Any other sequence is taken to be a sequence of separate indices to look up
                whole = self.whole
                domain = (whole[x] for x in domain_index)
        else:  # otherwise, we were given a non-slice non-iterable domain_index
            domain = self.whole[domain_index]
            domain_is_singular = True

        # -- If there weren't any range_indices we return this domain --
        if not range_indices:
            return domain if domain_is_singular else output_type(domain, nested=self._each_nested)

        # -- Otherwise we still need to index inside each member in the domain using the given range_indices --
        if len(range_indices) == 1: range_indices = range_indices[0]  # Treat 1-tuple as a single naked range index
        if domain_is_singular:  # If E[d] is a single member of E, E[d, range] returns its submember
            return domain[range_indices]
        if isinstance(domain, Mapping):  # If E[d] maps keys to members, E[d, range] maps keys to submembers
            return EachMapping({key: member[range_indices] for key, member in domain.items()})
        # Otherwise, E[d] is series of members, so E[d, range] is a series of submembers
        return output_type(member[range_indices] for member in domain)


    # E[...] = new_value should take different types depending on what indices are given, so we need @overload
    @overload  # E[index] = new_member replaces a single member
    def __setitem__(self, index:int, new_member:MemberType): ...
    @overload  # E[slice] = new_members replaces a slice of members with new members
    def __setitem__(self, slic:slice, new_members:Sequence[MemberType]): ...
    @overload  # E[index1, index2] = new_value sets m.[index2] = new_value, where m is E[index1]
    def __setitem__(self:'EachContainer[MemberType[SubmemberType]]', item:Tuple[int, int], value_to_broadcast:SubmemberType): ...
    @overload  # E[index, slice] = new_value sets m[slice] = new_value, where m is E[index1]
    def __setitem__(self:'EachContainer[MemberType[SubmemberType]]', item:Tuple[int, slice], value_to_broadcast:Sequence[SubmemberType]): ...
    @overload  # E[slice, index] = new_value sets m.[index] = new_value, for each m in E[slice]
    def __setitem__(self:'EachContainer[MemberType[SubmemberType]]', item:Tuple[slice, int], value_to_broadcast:SubmemberType): ...
    @overload  # E[slice1, slice2] = new_value sets m[slice2] = new_value, for each m in E[slice1]
    def __setitem__(self:'EachContainer[MemberType[SubmemberType]]', item:Tuple[slice, slice], value_to_broadcast:Sequence[SubmemberType]): ...
    def __setitem__(self, item, new_value):
        """`E[index] = new_value` replaces the designated value in EachContainer E.
           `E[slice] = new_values` replaces a slice with new members, much as doing this with a list would. If E is
            a mapping, keys between slice.start and slice.stop will have their values replaced, in order of key
            creation, with values iterated from new_values (or broadcast, if new_values is singular).
           `E[boolean_mask] = new_values` will replace values in E for whichever keys/indices the mask marks as True,
           in the mask's order, iterating from new_values as needed (or broadcasting new_values if it is singular).
           `each(C1)[mask] = each(C2)[mask]` will copy corresponding values that the mask marks as True from C2 to C1.
           `E[list_of_indices] = new_values` will set E[i] to a new_value for each i in list_of_indices, again
            iterating new values from new_values as needed (or broadcasting new_values if it is singular).
           `each(C1)[indices] = each(C2)[indices]` will copy corresponding values for the listed indices from C2 to C1.
           `E[domain, range] = new_value` sets m[range] = new_value, broadcast to all m in E[domain]. So,
           `E[:, range] = new_value` sets m[range] = new_value, broadcast to all members of E.
           TODO Note: the only case in which new_value would be distributed is E[:, range] = new_value, where range consists
            only of ints, as slicing may require a plural new_value to fill the slice, and E[index] can't vectorize."""
        #  TODO well, E[index] *can* vectorize now, at least for getting, though can't immediately support setting
        #   Stepping back and squinting:  setitem differs signficantly between the domain-only E[domain] = v case and
        #   the domain+range E[domain, range] = v case.  The domain+range case is relatively easy, because all we
        #   need to do is *get* the relevant domain members, and then pass the work of setting down to members.
        #   In contrast, setting domain-only requires altering E's own .whole.  That'll be fairly straightforward
        #   if that .whole offers relevant tools for setting, though when we've been agnostic about contents
        #   there may not be assurance of how many of these tools E.whole will support.
        #   If we want to support novel ways of setting that content, e.g. with boolean masks, we'll need a way of
        #   rearticulating that in a way that is understandable by E.whole.__setitem__.  This requires either
        #   that the content support setting [:] and inefficiently resetting all values, or that we can generate
        #   indices/keys just for each True part of the mask (e.g., with enumerate).
        #   This is somewhat related to challenges if we want to make each(dic).value[domain] = new_value work,
        #   which may also require some way of addressing items by dictionary keys.
        #

        # When setting E[domain, range] = v, we just need to identify the relevant domain members by getting E[domain],
        # and then set the relevant portion of each of those members to v, TODO broadcasting new values as needed
        if isinstance(item, tuple):  # E[domain, range] = new_value
            domain_index = item[0]                               # peel off the first index to slice the each-container itself
            range_index = item[1] if len(item)==2 else item[1:]  # remaining indices will slice each member
            for member, value in zip(self[domain_index], repeat_if_singular(new_value)):
                member[range_index] = value
            return

        # Otherwise we're dealing with the somewhat tricker case of E[domain] = v, where we need to tell E.whole to
        # modify itself, so it isn't enough to just get E[domain]. Instead we need to compute *indices* for E[domain].
        domain: Iterable[Union[int, KeyType]]  # will yield the keys/indices whose values in E should be replaced
        if isinstance(item, Mapping):  # Processing E[mapping] = v
            # A boolean mask mapping like E[E>1] would map keys/indices to True/False values
            if all(x is True or x is False for x in item.values()):
                domain = (key for key, boolean in item.items() if boolean is True)
            else:  # Processing E[mapping] where mapping.values() specify the keys/indices to replace
                domain = item.values()
        elif isinstance(item, Iterable) and not isinstance(item, str):  # Processing E[iterable] = v
            try:  # Check if the given item has introspectable length
                length = len(item)
            except TypeError:  # If it doesn't (e.g. due to being a raw generator) we'll spiel it into a tuple
                item = tuple(item)
            # A boolean mask sequence like E[E>1] would be a length-matched sequence of True/False values
            if len(item) == len(self.whole) and all(x is True or x is False for x in item):
                domain = (i for i, boolean in enumerate(item) if boolean is True)
            else:  # Any other sequence is taken to be a sequence of separate indices to modify
                domain = item
        elif isinstance(item, slice) and isinstance(self, Mapping):  # processing each(mapping)[slice] = v
            if item == slice(None, None, None):  # each(mapping)[:] = v replaces values for all keys
                domain = self.whole.keys()
            else:
                # TODO in theory, some Mappings could themselves support setitem with slice keys. Should try first?
                start, stop, step = item.start, item.stop, item.step
                domain = {key for key in self.whole.keys()
                              if (start is None or start <= key) and (stop is None or key < stop)}
        elif isinstance(item, slice) and (isinstance(new_value, str) or not isinstance(new_value, Iterable)):
            # processing each(sequence)[slice] = scalar; we need to compute indices of slice to stick scalar into
            domain = range(item.indices(len(self.whole)))
        else:  # otherwise, item is a scalar or a sequence-slice with an iterable new_value to splice into it
            self.whole[item] = new_value
            return
        # If we reach this point, domain is now an iterable of indices that we need to modify.
        for index, value in zip(domain, repeat_if_singular(new_value)):
            self.whole[index] = value

    # TODO this maybe should be able to have extra indices to reach down inside and surgerize within members???
    def __delitem__(self, item):
        del self.whole[item]

    # --- each-container distributing dunder methods across members ---

    def __getattr__(self, attr):  # motors.velocity returns each of m.velocity for all motors
        """E.attr returns each member's .attr, i.e. each(m.attr for m in E)."""
        return self._each_output_type(getattr(m, attr) for m in self.whole)

    def __setattr__(self, attr, value):  # E.attr = value sets each m.attr, broadcasting value if needed
        # with_matched_version of will broadcast other, but will not broadcast a length-1 self
        for m,v in with_matched_version_of(self, value):
            setattr(m, attr, v)

    def __call__(self, *args, **kwargs):  # E.method(args) calls m.method(args) for each member, broadcasting args
        # Note that E.method will create an EachContainer of bound methods, and then *its* __call__ will call them
        # We run together self, the args, and the values of kwargs for broadcasting, then disentangle after
        # TODO think about whether a class should be able to have an EachFunction as a method, and how to bind to instances
        B = BroadcastHandler(self, *args, *(kwargs.values()))
        start_k = 1 + len(args)  # fn takes 1, args take len(args), kwargs take the rest
        return B(f_a_k[0](*f_a_k[1:start_k], **{key: value for key, value in zip(kwargs.keys(), f_a_k[start_k:])})
                 for f_a_k in B)

    # TODO generalize return types and type-hints to allow subclasses like Pair to retain their (sub)class

    # --- each-container vectorized arithmetic ---
    # For type-hinting, we assume that each arithmetic op on a MemberType produces another MemberType

    # TODO technically should overload to show that dict+dict -> dict, whereas each+any->each
    def __add__(self, other)->'EachContainer[MemberType]':   # self + other
        B = BroadcastHandler(self, other)
        return B(s + o for s, o in B)

    # TODO generalize other ops to match __add__ so will return mappings for mappings

    def __radd__(self, other)->'EachContainer[MemberType]':  # other + self
        B = BroadcastHandler(self, other)
        return B(o + s for s, o in B)
    def __iadd__(self,other) -> 'EachContainer[MemberType]':  # self += other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s + o for s, o in B)
        # self.__dict__['whole'] = [s + o for s, o in with_matched_version_of(self, other)]
        # return self

    def __sub__(self, other)->'EachContainer[MemberType]':    # self - other
        B = BroadcastHandler(self, other)
        return B(s - o for s, o in B)
    def __rsub__(self, other)->'EachContainer[MemberType]':   # other - self
        B = BroadcastHandler(self, other)
        return B(o - s for s, o in B)
    def __isub__(self,other) -> 'EachContainer[MemberType]':  # self -= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s - o for s, o in B)

    def __mul__(self, other)->'EachContainer[MemberType]':    # self * other
        B = BroadcastHandler(self, other)
        return B(s * o for s, o in B)
    def __rmul__(self, other)->'EachContainer[MemberType]':   # other * self
        B = BroadcastHandler(self, other)
        return B(o * s for s, o in B)
    def __imul__(self,other) -> 'EachContainer[MemberType]':  # self *= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s * o for s, o in B)

    def __pow__(self, other)->'EachContainer[MemberType]':    # self ** other
        B = BroadcastHandler(self, other)
        return B(s ** o for s, o in B)
    def __rpow__(self, other)->'EachContainer[MemberType]':   # other ** self
        B = BroadcastHandler(self, other)
        return B(o ** s for s, o in B)
    def __ipow__(self,other) -> 'EachContainer[MemberType]':  # self **= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s ** o for s, o in B)

    def __mod__(self, other)->'EachContainer[MemberType]':    # self % other
        B = BroadcastHandler(self, other)
        return B(s % o for s, o in B)
    def __rmod__(self, other)->'EachContainer[MemberType]':   # other % self
        B = BroadcastHandler(self, other)
        return B(o % s for s, o in B)
    def __imod__(self,other) -> 'EachContainer[MemberType]':  # self %= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s % o for s, o in B)

    def __truediv__(self, other)->'EachContainer[MemberType]':    # self / other
        B = BroadcastHandler(self, other)
        return B(s / o for s, o in B)
    def __rtruediv__(self, other)->'EachContainer[MemberType]':   # other / self
        B = BroadcastHandler(self, other)
        return B(o / s for s, o in B)
    def __itruediv__(self,other) -> 'EachContainer[MemberType]':  # self /= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s / o for s, o in B)

    def __floordiv__(self, other)->'EachContainer[MemberType]':   # self // other
        B = BroadcastHandler(self, other)
        return B(s // o for s, o in B)
    def __rfloordiv__(self, other)->'EachContainer[MemberType]':  # other // self
        B = BroadcastHandler(self, other)
        return B(o // s for s, o in B)
    def __ifloordiv__(self,other) -> 'EachContainer[MemberType]':  # self //= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s // o for s, o in B)

    def __matmul__(self, other)->'EachContainer[MemberType]':    # self @ other
        B = BroadcastHandler(self, other)
        return B(s @ o for s, o in B)
    def __rmatmul__(self, other)->'EachContainer[MemberType]':   # other @ self
        B = BroadcastHandler(self, other)
        return B(o @ s for s, o in B)
    def __imatmul__(self,other) -> 'EachContainer[MemberType]':  # self @= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s @ o for s, o in B)

    # --- each-container vectorized bitwise ops ---
    def __and__(self, other)->'EachContainer[MemberType]':   # self & other
        B = BroadcastHandler(self, other)
        return B(s & o for s, o in B)
    def __rand__(self, other)->'EachContainer[MemberType]':  # other & self
        B = BroadcastHandler(self, other)
        return B(o & s for s, o in B)
    def __iand__(self,other) -> 'EachContainer[MemberType]':  # self &= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s & o for s, o in B)

    def __or__(self, other)->'EachContainer[MemberType]':    # self | other
        B = BroadcastHandler(self, other)
        return B(s | o for s, o in B)
    def __ror__(self, other)->'EachContainer[MemberType]':   # other | self
        B = BroadcastHandler(self, other)
        return B(o | s for s, o in B)
    def __ior__(self,other) -> 'EachContainer[MemberType]':  # self |= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s | o for s, o in B)

    def __xor__(self, other)->'EachContainer[MemberType]':    # self ^ other
        B = BroadcastHandler(self, other)
        return B(s ^ o for s, o in B)
    def __rxor__(self, other)->'EachContainer[MemberType]':   # other ^ self
        B = BroadcastHandler(self, other)
        return B(o ^ s for s, o in B)
    def __ixor__(self,other) -> 'EachContainer[MemberType]':  # self ^= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s ^ o for s, o in B)

    def __lshift__(self, other)->'EachContainer[MemberType]':    # self << other
        B = BroadcastHandler(self, other)
        return B(s << o for s, o in B)
    def __rlshift__(self, other)->'EachContainer[MemberType]':   # other << self
        B = BroadcastHandler(self, other)
        return B(o << s for s, o in B)
    def __ilshift__(self,other) -> 'EachContainer[MemberType]':  # self <<= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s << o for s, o in B)

    def __rshift__(self, other)->'EachContainer[MemberType]':    # self >> other
        B = BroadcastHandler(self, other)
        return B(s >> o for s, o in B)
    def __rrshift__(self, other)->'EachContainer[MemberType]':   # other >> self
        B = BroadcastHandler(self, other)
        return B(o >> s for s, o in B)
    def __irshift__(self,other) -> 'EachContainer[MemberType]':  # self >>= other
        B = BroadcastHandler(self, other, match_first=True)
        return B.in_place(s >> o for s, o in B)

    # --- each-container vectorized unary ops ---

    def __neg__(self)->'EachContainer[MemberType]':    # -self
        B = BroadcastHandler(self)
        return B(-s for (s,) in B)

    def __pos__(self)->'EachContainer[MemberType]':    # +self
        B = BroadcastHandler(self)
        return B(+s for (s,) in B)

    def __abs__(self)->'EachContainer[MemberType]':    # abs(self)
        B = BroadcastHandler(self)
        return B(abs(s) for (s,) in B)

    def __invert__(self)->'EachContainer[MemberType]':    # ~self
        B = BroadcastHandler(self)
        return B(~s for (s,) in B)

    def __round__(self, n=None) ->'EachContainer[MemberType]':  # round(self, n)
        B = BroadcastHandler(self)
        return B(round(s, n) for (s,) in B)

    def __trunc__(self) ->'EachContainer[MemberType]':  # math.trunc(self)
        B = BroadcastHandler(self)
        return B(trunc(s) for (s,) in B)

    def __floor__(self) ->'EachContainer[MemberType]':  # math.floor(self)
        B = BroadcastHandler(self)
        return B(floor(s) for (s,) in B)

    def __ceil__(self) ->'EachContainer[MemberType]':  # math.ceil(self)
        B = BroadcastHandler(self)
        return B(ceil(s) for (s,) in B)

    # --- each-container comparison operations ---

    def __gt__(self, other):
        B = BroadcastHandler(self, other)
        return B(s > o for s, o in B)

    def __ge__(self, other):
        B = BroadcastHandler(self, other)
        return B(s >= o for s, o in B)

    def __lt__(self, other):
        B = BroadcastHandler(self, other)
        return B(s < o for s, o in B)

    def __le__(self, other):
        B = BroadcastHandler(self, other)
        return B(s <= o for s, o in B)

EachContainer._each_output_type = EachContainer


class EachMapping(EachContainer[MemberType], MutableMapping[KeyType, MemberType]):
    """If D is a dictionary (or other MutableMapping), then each(D) will be an "each"-wrapped version of D that
       behaves much like D, but lends itself easily to distributed and vectorized operations involving D's values.
       Note that each-operations treat dictionaries as being containers of *values*, and their *keys* are just a more
       flexible way of indexing these values than the 0-based integer indices that lists and tuples use.
       (Alternatively it may help to view lists as inflexible dictionaries that always use 0...n as their "keys".)
       In vectorized operations involving each(D), items will be matched by key/index, so in general you'd want to do
       such operations only on dictionaries with matched keys and broadcastable singletons; or if D has integer keys,
       then each(D) can engage in vectorized operations with sequences, matching D's keys with sequence indices.
       Even if D does not have such integer keys, `each(D).values()` can engage in vectorized ops with sequences.
       `each(D1) + D2` creates an EachMapping mapping each of D1's keys to the sum of that key's values in D1 and D2.
       `each(D) + [3,4]` returns `each(D[0]+3, D[1]+4)`, or raises a KeyError if D doesn't have integer keys 0 and 1.
       `each(D).key and each(D).keys()` return an EachContainer with each of D's keys.
       `each(D).value and each(D).values()` return a sequence-like EachContainer with each of D's values.
       TODO `each(D).value = new_value` sets each key in D to the new_value (broadcasting where necessary)
       Note that `each(D1)|D2` ors each member of D1 with a member of D2, whereas `D1|D2` (in Python 3.9+)
       computes a union of whole dictionaries. This is analogous to `each(L1)+L2` adding each member, and `L1+L2`
       concatenating whole lists.  The whole corresponding to EachContainer E is `E.whole`.
       """
    # TODO It might also make sense to have other ops on dicts like each(math.sqrt)(D) make dicts with the same keys too

    # TODO consider whether `__contains__` and `__iter__` should involve *keys* as in traditional Python, or *values*
    #  as would be suggested by our analogy to lists

    """One approach would be to view each EachContainer as having a .index property that returns a (virtual) 
       EachContainer of indices for that container. For sequences, this'll basically be range(len(seq)); for dicts it'll 
       roughly be D.keys()  (though it may make sense to make this work like a mapping of each key to itself).
       This could be convenient for lots of other purposes, like doing math with sequence indices, explicitly looking 
       up items by index in some other like-indexed container, or making it easy to store indices for values of interest.
       Each each-op would somehow be responsible for ensuring that the output EachContainer gets an appropriate .index, 
       either inheriting a dictionary key structure, or using the vanilla one for sequences.
       
       Unfortunately there are lots of name collisions to worry about here.  List.index() is defined. Dict.keys() and 
       dict.values() are defined but perhaps could overload a __call__ method to pull a sort of double-duty.  
       Item.value often has another use.  Grammatically the singular each(D).key seems a bit more appropriate, and 
       a bit more analogous to each(D).attribute.  Since each(D) basically works as each(D).value anyway, there's 
       not that much to be gained by defining it, except as a learning crutch, and as a syntactic tool for getting 
       a version of D's values that vectorize sequence-like.
       
       each(D).value is sequence-like and matches items by position, rather than by key/index.  Does it remember keys? 
       
       A somewhat related puzzle is how we should think about each(set).
       We end up with an inconsistent pentad:
         1. dictionary *values* are like list members, since they are returned by indexing and needn't be unique.
         2. dictionary *keys* are like set members, since both must be unique and hashable
         3. set members are like list members, since this is the only plausible thing to iterate over
         4. likeness should be transitive
         5. dictionary keys are not like dictionary values.
       Traditionally Python has rejected 1, and has been reluctant to full-throatedly accept 2.
       My separate DictSet project fully embraces 2, viewing dicts as basically being sets that happen to have a value
       associated with each set member/key. This weighs somewhat against 1.
       I'm now entertaining a form of "multiple flavors" solution (accepting transitivity within flavors but not across). 
       Viewed as *wholes* dicts are set-like and should be able to engage in set-theoretic ops, like intersection 
       (rejects 1, accepts 2-5), and dicts just happen to have associated values.  From this perspective, lists are 
       deficient oddballs that require some slow/awkward special handling to get them to interact with the main players. 
       Viewed as *each-wise*aggregates*, dicts are list-like containers of values with idiosyncratic indices. From this 
       perspective, sets are deficient oddballs that fail to support any sort of indexing for their members.
       
       Might be useful to view sets as being an odd dictionary that maps each key to itself?    
       """

    whole: MutableMapping[KeyType, MemberType]

    # TODO since this is being defined just for typing purposes, perhaps can just use @overload???
    def __init__(self, whole: MutableMapping[KeyType, MemberType], nested: Union[int, bool, None] = 1):
        super().__init__(whole, nested = nested)

    def __repr__(self): return f"each({self.whole})"

    # The following is no longer needed since superclass version works; preserving docstring explanation
    # def __iter__(self) -> Iterable[MemberType]:
    #     """`for value in each(D)` iterates dictionary D's *values* not its *keys*, unlike how Python iterates D itself!
    #        This is because `each(X)` is intended to work as a container of indexable values, where vectorized
    #        operations match values by their indices, and X[i] returns the value associated with index i.
    #        When X is a list, the indices are integers, and the values returned by X[i] are list members.
    #        When X is a dictionary, the indices are keys, and the values returned by X[key] are dictionary values.
    #        Just as iterating each(L) yields each list value, and not the indices that L[i] maps to those values,
    #        iterating each(D) yields each dictionary value, and not the keys that D[key] maps to those values.
    #        Note that you can still iterate keys with `for key in `each(D).keys()`"""


    # # TODO reversed

    # TODO Check that Python doesn't mind having this done on values rather than keys?
    # Python always converts `m in C` checks to simple booleans, so we can't really distribute __contains__ itself
    def __contains__(self, item):
        """Returns true if item is equal to any of this EachMapping's *values*.  Note that this differs
           from ordinary Python dictionaries, which count their *keys* as being "in" them, rather than *values*.
           In general, EachMappings are construed as being likie lists with a more flexible indexing scheme,
           so just as 'item in L' checks if item matches a list-value rather than a list-index, `item in each(D)`
           checks if the item is among the EachMapping's values, rather than its keys/indices.
           You can use `item in EM.key` or `item in EM.whole` to check if an item is among EachMapping EM's keys.
           Note: Python requires that __contains__ return a single True/False value, so this is not distributed.
           If you want distributed containment checks, use E.contains(item) or E.is_in(container)."""
        return item in self.whole.values()

    # Python requires that bool return True or False, so no option to vectorize
    # To get a boolean for each key or each value in ED, use `each(bool)(ED.key)` or `each(bool)(ED)`
    def __bool__(self):
        return bool(self.whole)

    def keys(self) -> 'Collection[KeyType]':  # TODO is there a better type-hint for dict views?
        """If D is a dictionary, each(D).keys() returns D.keys(), a Python dictionary view of D's keys.
           Note: the property each(D).key is similar, but it returns an EachSet of the keys. This EachSet will
           engage in vectorized operations by matching keys, making it easier for such operations to interact with
           each(D) itself, e.g. in `each(D) + each(D).key`, whereas D.keys() is not an EachContainer so will not
           initiate vectorized operations, and if drawn into vectorized ops with an EachContainer it will match
           iteration order, making it easier for such operations to interact with other sequences, though iteration
           order will depend on the order of key-creation in D, and may change if a key is removed and re-added."""
        return self.whole.keys()

    @property
    def key(self) -> 'EachSet[KeyType]':
        """If D is a dictionary, each(D).key returns an EachSet of each key in D.
           Note: the method each(D).keys() is similar, but it returns a Python dictionary view of the keys. This EachSet
           will engage in vectorized operations by matching keys, making it easier for such operations to interact with
           each(D) itself, e.g. in `each(D) + each(D).key`, whereas D.keys() is not an EachContainer so will not
           initiate vectorized operations, and if drawn into vectorized ops with an EachContainer it will match
           iteration order, making it easier for such operations to interact with other sequences, though iteration
           order will depend on the order of key-creation in D, and may change if a key is removed and re-added."""
        return EachSet(self.whole.keys())

    def values(self) -> Collection[MemberType]:
        """If D is a dictionary, each(D).values() will be an EachContainer of each value in D, in order of key creation."""
        return self.whole.values()

    def items(self) -> EachContainer[Tuple[KeyType, MemberType]]:
        """If D is a dictionary, each(D).item and each(D).items() return an EachContainer of each item in D, as
           a (key, value) tuple, in order of key creation.  It is often easier to use each(D).key and each(D).value"""
        return EachContainer(self.whole.items())
    item = property(items)

    @overload
    def update(self, __m: Mapping[KeyType, MemberType], **kwargs: MemberType) -> None: ...
    @overload
    def update(self, __m: Iterable[Tuple[KeyType, MemberType]], **kwargs: MemberType) -> None: ...
    @overload
    def update(self, **kwargs: MemberType) -> None: ...
    def update(self, other, **kwargs): self.whole.update(other, **kwargs)

    def clear(self): self.whole.clear()
    def popitem(self): return self.whole.popitem()
    def copy(self:T) -> T:
        return type(self)(self.whole.copy())

    # TODO think about how to handle .get, .pop, .setdefault, and any custom methods a dict subclass might have
    #  These differ in that they accept a `key` argument that is sensibly distributable
    #  TODO think through whether each of these will automatically distribute in the right way?
    #  TODO consider defining expected ones anyway so that linter will know they're there, or will it due to ABC?

    # TODO think about the ._each_output_type for EachMapping. If it can be EachMapping this could inherit from superclass
    def __getattr__(self, attr):
        """each(D).attr creates a new EachMapping mapping each of D's keys to the corresponding value.attr.
           I.e. it distributes fetching .attr over each of D's values."""
        return EachMapping({key: getattr(value, attr) for key, value in self.whole.items()})


class EachSet(EachMapping[MemberType,MemberType]):
    """`each(S)` returns an EachSet for set S, which allows distributed operations on members of the set.
       In Python, sets generally do not have an assured ordering, so EachSets coordinate vectorized operations
       using their members simulataneously as keys and values, so can engage in vectorized operations with
       dictionaries with the same keys. An EachSet itself is (efficiently) implemented as though it is a trivial mapping
       of each member to itself, so each(S).keys() and each(S).values() are both S itself, and iterating each(S) yields
       its members. Since EachSets are effectively mappings, Vectorized operations on EachSets produce an EachMapping
       that maps each set member to the resulting value. Each of the following is a corollary.
       `each(S)*2` returns an EachMapping from each member of set S to twice its value.
       `each(S)*D` where D is a dictionary with matched keys, returns an EachMapping from each member m to m * D[m].
       `each(S).attr` returns an EachMapping from each member m to m.attr.
       `each(S).method(args)` returns an EachMapping from each member m to m.method(args), broadcasting args.
       `each(S)[member]` returns member's equivalent in S, if there is one, or raises an KeyError otherwise.
       `each(S)[each(S)>0]` returns the sub-EachSet that are positive
       `each(S)[list_of_keys]` returns an EachMapping from the keys to their equivalents in S (practically an EachSet)
       `each(D)[each(S)]` returns an EachMapping from each member m of S to its value D[m] in dictionary D.
       Note that so-called "set arithmetic" operations like `each(S) & X` will be distributed across each member of S,
       since that's what each() does! If you want an intersection of whole sets, use `S & X` or `each(S).whole & X`."""

    whole: Set

    def __init__(self, whole: Union[Set[MemberType],Mapping[MemberType,MemberType]],
                       nested: Union[int, bool, None] = next):
        if isinstance(whole, Mapping):
            whole = whole.keys()
        super().__init__(whole, nested=nested)

    @property
    def key(self) -> 'EachSet[MemberType, MemberType]':
        return self

    def keys(self) -> Collection[MemberType]:
        return self.whole

    def values(self) -> Collection[MemberType]:
        return self.whole

    def items(self) -> 'EachContainer[Tuple[MemberType, MemberType]]':
        return EachContainer(zip(self.whole, self.whole))

    # Override EachMapping superclass's manner of checking containment with simpler version available to sets
    def __contains__(self, item):
        return item in self.whole

    #TODO not sure if I need to repeat all the overloading?  Could just move this to a special case in superclass
    def __getitem__(self, item):
        if isinstance(item, slice) or (isinstance(item, Iterable) and not isinstance(item, str)):
            return super().__getitem__(item)
        if item in self.whole: return item
        raise KeyError(f"{item}")





