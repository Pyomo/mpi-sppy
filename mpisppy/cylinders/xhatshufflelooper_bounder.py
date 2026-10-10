###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import hashlib
import logging
import random
import mpisppy.log

from mpisppy import global_toc
from mpisppy.extensions.xhatbase import XhatBase
from mpisppy.cylinders.xhatbase import XhatInnerBoundBase
from mpisppy.cylinders._preloop_xhat_mixin import _PreLoopXhatMixin


# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.xhatshufflelooper_bounder",
                         "xhatclp.log",
                         level=logging.CRITICAL)
logger = logging.getLogger("mpisppy.cylinders.xhatshufflelooper_bounder")

class XhatShuffleInnerBound(_PreLoopXhatMixin, XhatInnerBoundBase):

    converger_spoke_char = 'X'

    def xhat_extension(self):
        return XhatBase(self.opt)

    def xhat_prep(self):
        self.xhatter = super().xhat_prep()

        ## option drive this? (could be dangerous)
        self.random_seed = 42
        # Have a separate stream for shuffling
        self.random_stream = random.Random()


    def try_scenario_dict(self, xhat_scenario_dict):
        """ wrapper for _try_one"""
        snamedict = xhat_scenario_dict

        stage2_ef_solver_name = self.opt.options.get("stage2_ef_solver_name", None)
        branching_factors = self.opt.options.get("branching_factors", None)  # for stage2ef
        obj = self.xhatter._try_one(snamedict,
                                    solver_options = self.solver_options,
                                    verbose=False,
                                    restore_nonants=True,
                                    stage2_ef_solver_name=stage2_ef_solver_name,
                                    branching_factors=branching_factors)
        def _vb(msg):
            if self.verbose and self.opt.cylinder_rank == 0:
                print ("(rank0) " + msg)

        if obj is None:
            _vb(f"    Infeasible {snamedict}")
            return False
        _vb(f"    Feasible {snamedict}, obj: {obj}")

        # XhatBase._try_one updates the solution cache in the opt object for us
        update = self.update_if_improving(obj, update_best_solution_cache=False)
        logger.debug(f'   bottom of try_scenario_dict on rank {self.global_rank}')
        return update

    def main(self):
        logger.debug(f"Entering main on xhatshuffle spoke rank {self.global_rank}")

        self.xhat_prep()

        # No-ops unless --xhatshuffle-try-jensens-first /
        # --xhatshuffle-try-feasible-xhat-first are set (mutually exclusive).
        # Both tolerate per-scenario infeasibility via silent skip.
        self._try_average_scenario_xhat()
        self._try_feasible_xhat()

        if "reverse" in self.opt.options["xhat_looper_options"]:
            self.reverse = self.opt.options["xhat_looper_options"]["reverse"]
        else:
            self.reverse = True
        if "iter_step" in self.opt.options["xhat_looper_options"]:
            self.iter_step = self.opt.options["xhat_looper_options"]["iter_step"]
        else:
            self.iter_step = None
        self.solver_options = self.opt.options["xhat_looper_options"]["xhat_solver_options"]

        # give all ranks the same seed
        self.random_stream.seed(self.random_seed)

        #We need to keep track of the way scenario_names were sorted
        scen_names = list(enumerate(self.opt.all_scenario_names))

        # shuffle the scenarios associated (i.e., sample without replacement)
        shuffled_scenarios = self.random_stream.sample(scen_names,
                                                       len(scen_names))

        # On self rather than local, so a checkpoint can reach them. The
        # cursor is where this spoke had got to in its exploration of the
        # scenarios -- where the next epoch starts, and its position and
        # direction in the order -- so a resumed spoke walks on the way the
        # uninterrupted one would rather than from the start. See
        # doc/designs/checkpointing_design.md section 5.6.
        self.scenario_cycler = ScenarioCycler(shuffled_scenarios,
                                              self.opt.nonleaves,
                                              self.reverse,
                                              self.iter_step)
        scenario_cycler = self.scenario_cycler
        self.xh_iter = 1
        #: The cursor this loop actually adopted, or None if it started fresh.
        #: Distinct from what the Checkpointer *read*: a cursor can be read
        #: and then refused (wrong scenario order), and a test that looked at
        #: the read would call that a success.
        self.applied_loop_state = None

        def _vb(msg):
            if self.verbose and self.opt.cylinder_rank == 0:
                print("(rank0) " + msg)

        # A resume hands back the cursor this spoke last checkpointed. It has
        # to happen here rather than in the Checkpointer's own restore hook:
        # that hook runs in pre_iter0, and the cycler does not exist until the
        # lines above. Same ordering as the hub's extension state.
        self._restore_loop_state_if_resuming()

        while not self.got_kill_signal():
            xh_iter = self.xh_iter
            # (unrelated: uncomment the next line to see the source of delay getting an xhat)
            if (xh_iter-1) % 100 == 0:
                logger.debug(f'   Xhatshuffle loop iter={xh_iter} on rank {self.global_rank}')
                logger.debug(f'   Xhatshuffle got from opt on rank {self.global_rank}')

            new_nonants = self.update_nonants()

            # When there is no iter0, the serial number must be checked.
            if self._nonant_len_receive_buffer.id() == 0:
                continue

            if new_nonants:
                # All cylinder_comm ranks agree on new_nonants because
                # update_nonants -> get_receive_buffer(synchronize=True)
                # gates on a cross-rank write_id Allreduce, so the
                # collectives inside this branch (Eobjective Allreduce,
                # comms["ROOT"].bcast in _try_one, the inner
                # got_kill_signal) are entered in lockstep.
                logger.debug(f'   *Xhatshuffle loop iter={xh_iter}')
                logger.debug(f'   *got a new one! on rank {self.global_rank}')
                logger.debug(f'   *localnonants={str(self.localnonants)}')

                # update the caches
                self.opt._put_nonant_cache(self.localnonants)
                # just for sending the values to other scenarios
                # so we don't need to tell persistent solvers
                self.opt._restore_nonants(update_persistent=False)

                _vb("   Begin epoch")
                scenario_cycler.begin_epoch()

                # always try at least two for each set of nonants
                # so we continue to explore the scenarios and
                # do not stall out on a single scenario because
                # the hub is moving very fast
                next_scendict = scenario_cycler.get_next()
                if next_scendict is not None:
                    _vb(f"   Trying next {next_scendict}")
                    update = self.try_scenario_dict(next_scendict)
                    if update:
                        _vb(f"   Updating best to {next_scendict}")
                        scenario_cycler.best = next_scendict["ROOT"]

                if self.got_kill_signal():
                    # time to go; don't solve next -- but the try just above
                    # may have improved the incumbent, and this exit skips
                    # the bottom of the loop.
                    self.maybe_checkpoint()
                    return

            next_scendict = scenario_cycler.get_next()
            if next_scendict is not None:
                _vb(f"   Trying next {next_scendict}")
                update = self.try_scenario_dict(next_scendict)
                if update:
                    _vb(f"   Updating best to {next_scendict}")
                    scenario_cycler.best = next_scendict["ROOT"]

            #_vb(f"    scenario_cycler._scenarios_this_epoch {scenario_cycler._scenarios_this_epoch}")

            self.maybe_checkpoint()

            self.xh_iter += 1

    def checkpoint_loop_state(self):
        """This spoke's place in its own loop, for the checkpoint file.

        The scenario order itself is not carried: the shuffle is seeded to a
        fixed value and drawn once, so a resumed spoke reproduces it exactly.
        What cannot be reproduced is how far through it this spoke had got --
        that depends on how many passes it managed before the stop, which
        depends on the hub.
        """
        cycler = getattr(self, "scenario_cycler", None)
        if cycler is None:
            # Asked before main() built the loop -- there is no position yet.
            return None
        # Written from the bottom of the pass, so this is the pass that just
        # completed -- the same thing the hub's generation number means.
        return {"xh_iter": int(self.xh_iter),
                "cursor": cycler.checkpoint_state()}

    def loop_state_progress(self, state):
        """Only the cursor. xh_iter counts passes, not work.

        The loop increments xh_iter at the bottom of every pass, including the
        ones that only poll the hub and solve nothing -- it feeds the periodic
        debug line about where the delay in getting an xhat comes from, so it
        has to count those. The cursor moves only when a subproblem was
        actually solved, which is the thing worth a write.
        """
        return state.get("cursor") if isinstance(state, dict) else None

    def restore_loop_state(self, state):
        """Put the cursor back where the checkpoint left it.

        Returns a list of warnings rather than raising: a cursor that no
        longer fits the run is a reason to explore from the start again, not
        a reason to throw away the incumbent in the same file and refuse the
        resume.
        """
        missing = [key for key in ("cursor", "xh_iter")
                   if not isinstance(state, dict) or key not in state]
        if missing:
            return [f"the checkpointed xhatshuffle loop state has no "
                    f"{', '.join(missing)}, so this spoke explores from the "
                    f"start again."]
        xh_iter = state["xh_iter"]
        if (not isinstance(xh_iter, int) or isinstance(xh_iter, bool)
                or xh_iter < 0):
            return ["the checkpointed xhatshuffle loop state has a bad value "
                    "for xh_iter, so this spoke explores from the start "
                    "again."]
        warnings = self.scenario_cycler.restore_state(state["cursor"])
        if not warnings:
            # The file records the pass that completed, so the resumed loop
            # starts at the next one -- the same convention the hub uses for
            # its iteration counter, and it keeps pass numbers in the log
            # unique across a stop.
            self.xh_iter = int(state["xh_iter"]) + 1
        return warnings

    def _restore_loop_state_if_resuming(self):
        """Ask the Checkpointer for a restored cursor, if this is a resume.

        The ranks of a multi-rank spoke walk this cursor together, so they
        have to adopt the same one. They do, and not by luck: the Checkpointer
        agreed the cursor across them when it read the files
        (``checkpointing.agree_spoke_restore``), and everything below is a
        function of that cursor and of the scenario order, which is drawn from
        one seed and so is identical on every rank. Do not make this a
        rank-local decision again.
        """
        state = self._checkpointed_loop_state()
        if state is None:
            return
        warnings = self.restore_loop_state(state)
        for message in warnings:
            global_toc(f"WARNING: {message}", self.opt.cylinder_rank == 0)
        if not warnings:
            self.applied_loop_state = state
            # The first pass after a resume always begins a new epoch (the
            # hub's nonants are new to this run), and an epoch starts from
            # the best scenario so far, so that is what is tried next.
            best = self.scenario_cycler.best
            global_toc(
                f"Restored the checkpointed xhatshuffle cursor "
                f"(pass {self.xh_iter}; the next epoch starts from "
                f"{'the start of the scenario order' if best is None else best})",
                self.opt.cylinder_rank == 0)


class ScenarioCycler:

    def __init__(self, shuffled_scenarios,nonleaves,reverse,iter_step):
        root_kids = nonleaves['ROOT'].kids if 'ROOT' in nonleaves else None
        if root_kids is None or len(root_kids)==0 or root_kids[0].is_leaf:
            self._multi = False
            self._iter_shift = 0 if iter_step is None else iter_step
            self._use_reverse = False #It is useless to reverse for 2stage SP
        else:
            self._multi = True
            self.BF0 = len(root_kids)
            self._nonleaves = nonleaves

            # TODO: is this right for multistage, or should the default be
            #       0 like in the two-stage case?
            self._iter_shift = self.BF0 if iter_step is None else iter_step
            self._use_reverse = True if reverse is None else reverse
            self._reversed = False #Do we iter in reverse mode ?
        self._shuffled_scenarios = shuffled_scenarios
        self._num_scenarios = len(shuffled_scenarios)

        self._cycle_idx = 0
        self._best = None
        self._begin_normal_epoch()


    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        self._best = value

    def _order_fingerprint(self):
        """Identify the scenario order these indices are indices *into*.

        The cursor is an index, so it only means anything against the order it
        was taken from. That order is deterministic -- the shuffle is seeded to
        a fixed value and drawn once from ``all_scenario_names`` -- so a resumed
        spoke reproduces it, and this is what proves it did. If the model's
        scenario list changed, index 7 is a different scenario and restoring it
        would silently send the spoke somewhere else.

        A hash rather than the list itself: this rides in a file that is
        rewritten whenever the cursor moves, and a run with many thousands of
        scenarios should not pay for a copy of every name each time.
        """
        blob = "\n".join(f"{i}:{name}" for i, name in self._shuffled_scenarios)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cursor_problem(self, state):
        """What is wrong with a cursor of the current format, or None.

        A file of the current format version can still be hand-edited, or
        written before a change to checkpoint_state() that did not raise the
        version, so the keys and the types of the values are checked rather
        than assumed.
        """
        missing = sorted(set(self.checkpoint_state()) - set(state))
        if missing:
            return f"has no {', '.join(missing)}"

        def is_name(value):
            return value is None or isinstance(value, str)

        idx = state["cycle_idx"]
        bad = []
        if (not isinstance(idx, int) or isinstance(idx, bool)
                or not 0 <= idx < self._num_scenarios):
            bad.append("cycle_idx")
        if not is_name(state["best"]):
            bad.append("best")
        if not is_name(state["cur_root_scen"]):
            bad.append("cur_root_scen")
        tried = state["scenarios_this_epoch"]
        if (not isinstance(tried, (list, tuple))
                or not all(isinstance(n, str) for n in tried)):
            bad.append("scenarios_this_epoch")
        if not isinstance(state["reversed"], bool):
            bad.append("reversed")
        nodes = state["nodescen_dict"]
        if (not isinstance(nodes, dict)
                or not all(isinstance(k, str) and is_name(v)
                           for k, v in nodes.items())):
            bad.append("nodescen_dict")
        if bad:
            return f"has a bad value for {', '.join(bad)}"

        # Well formed, but does it describe this run? A name that is not one
        # of this run's scenarios would reach _try_one as a scenario to
        # evaluate, and a node set that is not this tree's would leave a
        # node with no scenario to try.
        known = set(self._shuffled_snames)
        names = [state["best"], state["cur_root_scen"],
                 *state["scenarios_this_epoch"], *nodes.values()]
        unknown = sorted({n for n in names if n is not None and n not in known})
        if unknown:
            return f"names scenarios this run does not have ({unknown[:3]})"
        # _create_nodescen_dict fills every nonleaf node each epoch, or just
        # ROOT for a two-stage problem.
        expected = set(self._nonleaves) if self._multi else {"ROOT"}
        if set(nodes) != expected:
            return (f"covers tree nodes {sorted(nodes)}, not this run's "
                    f"{sorted(expected)}")
        return None

    def checkpoint_state(self):
        """Where this cycler has got to, as plain data.

        Everything derived from ``_shuffled_scenarios`` is left out and
        rebuilt on restore, because the scenario order is reproduced exactly
        rather than carried.
        """
        return {
            "order_fingerprint": self._order_fingerprint(),
            "cycle_idx": self._cycle_idx,
            "best": self._best,
            "cur_root_scen": self._cur_ROOTscen,
            # A set is not worth the round trip; order does not matter to the
            # membership tests that read it.
            "scenarios_this_epoch": sorted(self._scenarios_this_epoch),
            "reversed": getattr(self, "_reversed", False),
            "nodescen_dict": dict(self.nodescen_dict),
        }

    def restore_state(self, state):
        """Put the cursor back. Returns a list of warnings; [] on success.

        A cursor that does not fit this run is discarded rather than raising:
        the same file carries the incumbent, which is the part worth keeping,
        and exploring from the start again is a cost rather than an error.
        """
        if not isinstance(state, dict):
            return ["the checkpointed xhatshuffle cursor is not readable, so "
                    "this spoke explores from the start again."]
        if state.get("order_fingerprint") != self._order_fingerprint():
            return ["the checkpointed xhatshuffle cursor was taken against a "
                    "different scenario order, so its position means nothing "
                    "here; this spoke explores from the start again."]
        # Checked before anything is changed, so a damaged cursor is
        # discarded whole rather than half applied.
        problem = self._cursor_problem(state)
        if problem is not None:
            return [f"the checkpointed xhatshuffle cursor {problem}, so this "
                    f"spoke explores from the start again."]

        # `best` first: the epoch rebuild below reads it to decide where the
        # epoch starts. The position is overwritten afterwards either way, but
        # having the rebuild see the real value keeps this a restore rather
        # than a sequence of corrections.
        self._best = state["best"]

        # Then re-derive the epoch's view of the order: _begin_*_epoch is what
        # sets _shuffled_snames/_original_order, and the two differ by
        # direction, so the direction has to be applied before the position.
        if state["reversed"]:
            self._begin_reverse_epoch()
        else:
            self._begin_normal_epoch()

        self._cycle_idx = state["cycle_idx"]
        self._cur_ROOTscen = state["cur_root_scen"]
        self._scenarios_this_epoch = set(state["scenarios_this_epoch"])
        self.nodescen_dict = dict(state["nodescen_dict"])
        return []

    def _fill_nodescen_dict(self,empty_nodes):
        filling_idx = self._cycle_idx
        while len(empty_nodes) >0:
            #Sanity check to make no infinite loop.
            if filling_idx == self._cycle_idx and 'ROOT' in self.nodescen_dict and self.nodescen_dict['ROOT'] is not None:
                print(self.nodescen_dict)
                raise RuntimeError("_fill_nodescen_dict looped over every scenario but was not able to find a scen for every nonleaf node.")
            sname = self._shuffled_snames[filling_idx]
            snum = self._original_order[filling_idx]

            def _add_sname_to_node(ndn):
                first = self._nonleaves[ndn].scenfirst
                last = self._nonleaves[ndn].scenlast
                if snum>=first and snum<=last:
                    self.nodescen_dict[ndn] = sname
                    return False
                else:
                    return True
            #Adding sname to every nodes it goes by, and removing the nodes from empty_nodes
            empty_nodes = list(filter(_add_sname_to_node,empty_nodes))
            filling_idx +=1
            filling_idx %= self._num_scenarios

    def _create_nodescen_dict(self):
        '''
        Creates an attribute nodescen_dict.
        Keys are nonleaf names, values are local scenario names
        (a value can be None if the associated scenario is not in our rank)

        WARNING: _cur_ROOTscen must be up to date when calling this method
        '''
        if not self._multi:
            self.nodescen_dict = {'ROOT':self._cur_ROOTscen}
        else:
            self.nodescen_dict = dict()
            self._fill_nodescen_dict(self._nonleaves.keys())

    def _update_nodescen_dict(self,snames_to_remove):
        '''
        WARNING: _cur_ROOTscen must be up to date when calling this method
        '''
        if not self._multi:
            self.nodescen_dict = {'ROOT':self._cur_ROOTscen}
        else:
            empty_nodes = []
            for ndn in self._nonleaves.keys():
                if self.nodescen_dict[ndn] in snames_to_remove:
                    self.nodescen_dict[ndn] = None
                    empty_nodes.append(ndn)
            self._fill_nodescen_dict(empty_nodes)


    def begin_epoch(self):
        if self._multi and self._use_reverse and not self._reversed:
            self._begin_reverse_epoch()
        else:
            self._begin_normal_epoch()

    def _begin_normal_epoch(self):
        if self._multi:
            self._reversed = False
        self._shuffled_snames = [s[1] for s in self._shuffled_scenarios]
        self._original_order = [s[0] for s in self._shuffled_scenarios]
        self._cur_ROOTscen = self._shuffled_snames[0] if self.best is None else self.best
        self._create_nodescen_dict()
        self._scenarios_this_epoch = set()

    def _begin_reverse_epoch(self):
        self._reversed = True
        self._shuffled_snames = [s[1] for s in reversed(self._shuffled_scenarios)]
        self._original_order = [s[0] for s in reversed(self._shuffled_scenarios)]
        self._cur_ROOTscen = self._shuffled_snames[0] if self.best is None else self.best
        self._create_nodescen_dict()
        self._scenarios_this_epoch = set()

    def get_next(self):
        next_scen = self._cur_ROOTscen
        next_scendict = self.nodescen_dict
        if next_scen in self._scenarios_this_epoch:
            return None
        self._scenarios_this_epoch.add(next_scen)
        self._iter_scen()
        return next_scendict

    def _iter_scen(self):
        old_idx = self._cycle_idx
        self._cycle_idx += self._iter_shift
        ## wrap around
        self._cycle_idx %= self._num_scenarios

        #do not reuse a previously visited scenario for 'ROOT'
        tmp_cycle_idx = self._cycle_idx
        while self._shuffled_snames[tmp_cycle_idx] in self._scenarios_this_epoch and (
                (tmp_cycle_idx+1)%self._num_scenarios != self._cycle_idx):
            tmp_cycle_idx +=1
            tmp_cycle_idx %= self._num_scenarios

        self._cycle_idx = tmp_cycle_idx

        #Updating scenarios
        self._cur_ROOTscen = self._shuffled_snames[self._cycle_idx]
        if old_idx<self._cycle_idx:
            scens_to_remove = set(self._shuffled_snames[old_idx:self._cycle_idx])
        else:
            scens_to_remove = set(self._shuffled_snames[old_idx:]+self._shuffled_snames[:self._cycle_idx])
        self._update_nodescen_dict(scens_to_remove)
