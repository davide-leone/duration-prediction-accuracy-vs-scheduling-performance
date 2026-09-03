# """
# MIT License

# Copyright (c) 2017 cgalleguillosm, AlessioNetti

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# """
import logging

from sys import maxsize
from random import seed
from abc import abstractmethod, ABC
from sortedcontainers.sortedlist import SortedListWithKey
from enum import Enum
from copy import deepcopy

from accasim.base.resource_manager_class import ResourceManager 
from accasim.base.allocator_class import AllocatorBase


class DispatcherError(Exception):
    pass


class JobVerification(Enum):
    
    REJECT = -1  # All jobs are rejected
    NO_CHECK = 0  # No verification
    CHECK_TOTAL = 1  # Total requested resources are verified  
    CHECK_REQUEST = 2  # Each node x resources are verified


class SchedulerBase(ABC):
    
    """
    
        This class allows to implement dispatching methods by integrating with an implementation of this class an allocator (:class:`accasim.base.allocator_class.AllocatorBase`). 
        An implementation of this class could also serve as a entire dispatching method if the allocation class is not used as default (:class:`.allocator` = None), but the resource manager must
        be set on the allocator using :func:`accasim.base.allocator_class.AllocatorBase.set_resource_manager`.
        
    """
    MAXSIZE = maxsize
    ALLOW_MAPPING_SAME_NODE = True
    
    def __init__(self, _seed, allocator=None, job_check=JobVerification.CHECK_REQUEST, **kwargs):
        """
        
        Construct a scheduler
            
        :param seed: Seed for the random state
        :param resource_manager: A Resource Manager object for dealing with system resources.
        :param allocator: Allocator object to be used by the scheduler to allocater after schedule generation. If an allocator isn't defined, the scheduler class must generate the entire dispatching plan.
        :param job_check: A job may be rejected if it doesnt comply with:
                    - JobVerification.REJECT: Any job is rejected
                    - JobVerification.NO_CHECK: All jobs are accepted
                    - JobVerification.CHECK_TOTAL: If the job requires more resources than the available in the system.
                    - JobVerification.CHECK_REQUEST: if an individual request by node requests more resources than the available one.
                    
                    
        :param kwargs:
            - skip_jobs_on_allocation: If the allocator is predefined and this parameter is true, the allocator will try to allocate jobs as much as possible. 
                Otherwise, the allocation will stop after the first fail.
                
        """
        seed(_seed)
        self._counter = 0
        self.allocator = None
        self._logger = logging.getLogger('accasim')
        self._system_capacity = None
        self._nodes_capacity = None
        self.resource_manager = None
                
        if allocator:
            assert isinstance(allocator, AllocatorBase), 'Allocator not valid for scheduler'
            self.allocator = allocator
        # self.set_resource_manager(resource_manager)

        assert(isinstance(job_check, JobVerification)), 'job_check invalid type. {}'.format(job_check.__class__)
        if job_check == JobVerification.REJECT:
            print('All jobs will be rejected, and for performance purposes the rejection messages will be omitted.')
        self._job_check = job_check
        
        # Check resources
        self._min_required_availability = kwargs.pop('min_resources', None)  # ['core', 'mem']s
        # Skip jobs during allocation
        self.skip_jobs_on_allocation = kwargs.pop('skip_jobs_on_allocation', False)
                
        
    @property
    def name(self):
        """
        
        Name of the schedulign method
        
        """
        raise NotImplementedError 
    
    @abstractmethod
    def get_id(self):
        """
        
        Must return the full ID of the scheduler, including policy and allocator.
        
        :return: the scheduler's id.
        
        """
        raise NotImplementedError
    
    @abstractmethod
    def scheduling_method(self, cur_time, es_dict, es):
        """
        
        This function must map the queued events to available nodes at the current time.
            
        :param cur_time: current time
        :param es_dict: dictionary with full data of the job events
        :param es: events to be scheduled
            
        :return a tuple of (time to schedule, event id, list of assigned nodes), an array jobs id of rejected jobs  
        
        """
        raise Exception('This function must be implemented!!')
    
    def set_resource_manager(self, resource_manager):
        """
        
        Set a resource manager. 

        :param resource_manager: An instantiation of a resource_manager class or None 
        
        """        
        if resource_manager:
            if self.allocator:
                self.allocator.set_resource_manager(resource_manager)
            assert isinstance(resource_manager, ResourceManager), 'Resource Manager not valid for scheduler'
            self.resource_manager = resource_manager
        else:
            self.resource_manager = None
            
    def schedule(self, cur_time, es_dict, es):
        """
        
        Method for schedule. It calls the specific scheduling method.
        
        :param cur_time: current time
        :param es_dict: dictionary with full data of the events
        :param es: events to be scheduled
        
        :return: a tuple of (time to schedule, event id, list of assigned nodes), array of rejected job ids.
        
        """
        assert(self.resource_manager is not None), 'The resource manager is not defined. It must defined prior to run the simulation.'

        self._counter += 1
        self._logger.debug("{} Dispatching: #{} decision".format(cur_time, self._counter))
        self._logger.debug('{} Dispatching: {} queued jobs'.format(cur_time, len(es)))
        self._logger.debug('{} Dispatching: {}'.format(cur_time, self.resource_manager.current_usage))

        rejected = []
        
        # At least a job need 1 core and 1 kb/mb/gb of mem to run
        if self._min_required_availability and any([self.resource_manager.resources.full[res] for res in self._min_required_availability]):
            self._logger.debug("There is no availability of one of the min required resource to run a job. The dispatching process will be delayed until there is enough resources.")
            return [(None, e, []) for e in es], rejected

        accepted = []
        # Verify jobs with the defined Job Policy
        for e in es:
            job = es_dict[e]
            if not job.get_checked() and not self._check_job_request(job):
                if self._job_check != JobVerification.REJECT:
                    self._logger.warning('{} has been rejected by the dispatcher. ({})'.format(e, self._job_check))
                rejected.append(e)
                continue
            accepted.append(job)
            
        to_allocate = []
        # On accepted jobs by policy, try to schedule with the scheduling policy
        if accepted:
            to_allocate, to_reject = self.scheduling_method(cur_time, accepted, es_dict)
            rejected += to_reject
            for e in to_reject:
                self._logger.warning('{} has been rejected by the dispatcher. (Scheduling policy)'.format(e))         
        
        # If there are scheduled jobs and an allocator defined, try to allocate the scheduled jobs. 
        if to_allocate and self.allocator:
            dispatching_plan = self.allocator.allocate(to_allocate, cur_time, skip=self.skip_jobs_on_allocation)
        else:
            dispatching_plan = to_allocate
            
        return dispatching_plan, rejected
    
    def _check_job_request(self, _job):
        """

        Simple method that checks if the loaded _job violates the system's resource constraints.

        :param _job: Job object

        :return: True if the _job is valid, false otherwise

        """
        _job.set_checked(True)
        if self._job_check == JobVerification.REJECT:
            return False
        
        elif self._job_check == JobVerification.NO_CHECK:
            return True
        
        elif self._job_check == JobVerification.CHECK_TOTAL:
            # We verify that the _job does not violate the system's resource constraints by comparing the total
            if not self._system_capacity:
                self._system_capacity = self.resource_manager.system_capacity('total')
            return not any([_job.requested_resources[res] * _job.requested_nodes > self._system_capacity[res] for res in _job.requested_resources.keys()])
                
        elif self._job_check == JobVerification.CHECK_REQUEST:
            if not self._nodes_capacity:
                self._nodes_capacity = self.resource_manager.system_capacity('nodes')
            # We verify the _job request can be fitted in the system        
            _requested_resources = _job.requested_resources
            _requested_nodes = _job.requested_nodes

            _fits = 0
            _diff_node = 0 
            for _node, _attrs in self._nodes_capacity.items():
                # How many time a request fits on the node
                _nfits = min([_attrs[_attr] // req for _attr, req in _requested_resources.items() if req > 0 ])
                # Update current number of times the current job fits in the nodes
                if _nfits > 0:
                    _fits += _nfits
                    _diff_node += 1
                    
                if self.ALLOW_MAPPING_SAME_NODE:
                    # Since _fits >> _diff_node this logical comparison is omitted.
                    if _fits >= _requested_nodes: 
                        return True
                else:
                    if _diff_node >= _requested_nodes:
                        return True
            
            return False
        raise DispatcherError('Invalid option.')    
    
    def __str__(self):
        return self.get_id()


class SimpleHeuristic(SchedulerBase):
    """
    
    Simple scheduler, sorts the event depending on the chosen policy.
    
    If a single job allocation fails, all subsequent jobs fail too.
    Sorting as name, sort funct parameters
    
    """

    def __init__(self, seed, allocator, name, sorting_parameters, **kwargs):
        SchedulerBase.__init__(self, seed, allocator, **kwargs)
        self.name = name
        self.sorting_parameters = sorting_parameters

    def get_id(self):
        """
        
        Returns the full ID of the scheduler, including policy and allocator.

        :return: the scheduler's id.
        
        """
        return '-'.join([self.__class__.__name__, self.name, self.allocator.get_id()])

    def scheduling_method(self, cur_time, jobs, es_dict):
        """
        
        This function must map the queued events to available nodes at the current time.
        
        :param cur_time: current time
        :param es_dict: dictionary with full data of the events
        :param es: events to be scheduled
        
        :return: a tuple of (time to schedule, event id, list of assigned nodes), an array jobs id of rejected jobs  
        
        """
        to_reject = []
               
        to_schedule = SortedListWithKey(jobs, **self.sorting_parameters)
        return to_schedule, to_reject


class FirstInFirstOut(SimpleHeuristic):
    """

    **FirstInFirstOut scheduling policy.** 
    
    The first come, first served (commonly called FirstInFirstOut ‒ first in, first out) 
    process scheduling algorithm is the simplest process scheduling algorithm. 
        
    """
    name = 'FIFO'
    """ Name of the Scheduler policy. """
    
    sorting_arguments = {
            'key': lambda x: x.queued_time
        }
    """ This sorting function allows to sort the jobs in relation of the scheduling policy. """

    def __init__(self, _allocator, _seed=0, **kwargs):
        """
        
        FirstInFirstOut Constructor
        
        """
        SimpleHeuristic.__init__(self, _seed, _allocator, self.name, self.sorting_arguments, **kwargs)


class LongestJobFirst(SimpleHeuristic):
    """
    
    **LJF scheduling policy.**
    
    Longest Job First (LJF) sorts the jobs, where the longest jobs are preferred over the shortest ones.  
        
    """
    name = 'LJF'
    """ Name of the Scheduler policy. """
    
    sorting_arguments = {
            'key': lambda x:-x.expected_duration
        }
    """ This sorting function allows to sort the jobs in relation of the scheduling policy. """

    def __init__(self, _allocator, _resource_manager=None, _seed=0, **kwargs):
        """
        
        LJF Constructor
        
        """
        SimpleHeuristic.__init__(self, _seed, _allocator, self.name, self.sorting_arguments, **kwargs)


class ShortestJobFirst(SimpleHeuristic):
    """
    
    **SJF scheduling policy.**
    
    Shortest Job First (SJF) sorts the jobs, where the shortest jobs are preferred over the longest ones.
    
    """
    name = 'SJF'
    """ Name of the Scheduler policy. """
    
    sorting_arguments = {
            'key': lambda x: x.expected_duration
        }
    """ This sorting function allows to sort the jobs in relation of the scheduling policy. """

    def __init__(self, _allocator, _resource_manager=None, _seed=0, **kwargs):
        """
    
        SJF Constructor
    
        """
        SimpleHeuristic.__init__(self, _seed, _allocator, self.name, self.sorting_arguments, **kwargs)


class EASYBackfilling(SchedulerBase):
    """
   
   EASY Backfilling scheduler.
   
   Whenever a job cannot be allocated, a reservation is made for it. After this, the following jobs are used to
   backfill the schedule, not allowing them to use the reserved nodes.
     
   This dispatching methods includes its own calls to the allocator over the dispatching process.
   Then it isn't use the auto allocator call, after the schedule generation.    
   
   """
   
    name = 'EBF'
    """ Name of the Scheduler policy. """
       
    def __init__(self, allocator, seed=0, **kwargs):
        """
   
       Easy BackFilling Constructor
      
       """
        SchedulerBase.__init__(self, seed, allocator=None, **kwargs)
        self._blocked_job_id = None
        self._reserved_slot = (None, [],)
        self.nonauto_allocator = allocator
        self.allocator_rm_set = False
        # self.nonauto_allocator.set_resource_manager(resource_manager)
       
    def get_id(self):
        """
   
       Returns the full ID of the scheduler, including policy and allocator.
       :return: the scheduler's id.
   
       """
        return '-'.join([self.name, self.nonauto_allocator.name])
 
    def scheduling_method(self, cur_time, queued_jobs, es_dict):
        """
        This function must map the queued events to available nodes at the current time.
       
        :param cur_time: current time
        :param queued_jobs: Jobs to be dispatched
        :param es_dict: dictionary with full data of the events
        
        
        :return: a list of tuples (time to schedule, event id, list of assigned nodes), and a list of rejected job ids  
        """
        if not self.allocator_rm_set:
            self.nonauto_allocator.set_resource_manager(self.resource_manager)
            self.allocator_rm_set = True   

                   
        avl_resources = self.resource_manager.current_availability
        self.nonauto_allocator.set_resources(avl_resources)
               
        to_dispatch = []
        to_reject = []
        _to_fill = []
        _prev_blocked = None
        _time_reached = False
        
        if self._reserved_slot[0] and self._reserved_slot[0] <= cur_time:
            _time_reached = True 
            # Tries to allocate the blocked job
            self._logger.trace('There is a blocked job {} with {}'.format(self._blocked_job_id, self._reserved_slot))
            # assert(self._blocked_job_id == queued_jobs[0].id), 'The first element is not the blocked one. ({} != {})'.format(self._blocked_job_id, queued_jobs[0].id)

            blocked_job = queued_jobs[0]
            queued_jobs = queued_jobs[1:]
                        
            allocation = self.nonauto_allocator.allocating_method(blocked_job, cur_time, skip=False)
                
            if allocation[-1]:
                self._logger.trace('{}: {} blocked job can be allocated. Unblocking'.format(cur_time, self._blocked_job_id))
                self._blocked_job_id = None
                self._reserved_slot = (None, [])
                _prev_blocked = [allocation]
                    
            else:
                # There are jobs still using the reserved nodes           
                self._logger.trace('{} job is still blocked. Reservation {}'.format(self._blocked_job_id, self._reserved_slot))
            # Add the current allocation for the (un)blocked job.
            to_dispatch += [allocation]
        
        if self._blocked_job_id is None and queued_jobs:
            # Tries to perform a FIFO allocation if there is no blocked job 
            # Returns the (partial) allocation and the idx for the blocked job, also sets the self._blocked_job_id var
            _allocated_jobs, blocked_idx = self._try_fifo_allocation(queued_jobs, cur_time)

            # There is a blocked job
            if not (blocked_idx is None):
                # If there is no a reservation, calculate it for the blocked job
                if not self._reserved_slot[0]:
                    blocked_job = queued_jobs[blocked_idx]
                    self._logger.trace('Blocked {} Job: Calculate the reservation'.format(self._blocked_job_id))
                   
                    # Current reservation (future time, reserved nodes)
                    self._reserved_slot = self._calculate_slot(cur_time, deepcopy(avl_resources), _allocated_jobs[:blocked_idx], _prev_blocked, blocked_job, es_dict)
                    self._logger.trace('Blocked {} Job: Nodes {} are reserved at {}'.format(self._blocked_job_id, self._reserved_slot[1], self._reserved_slot[0]))
                
                # Include the blocked job                
                to_dispatch += _allocated_jobs[:blocked_idx + 1]
                _to_fill = queued_jobs[blocked_idx + 1:]
            else:
                to_dispatch += _allocated_jobs                    
        else:
            if not _time_reached:
                # The blocked job
                to_dispatch += [(None, self._blocked_job_id, [])]
                # All the remaining queued jobs
                _to_fill = queued_jobs[1:]
            else:
                # The remaining queued jobs
                _to_fill = queued_jobs
        
        if _to_fill:
            self._logger.trace('Blocked job {}. {} jobs candidates to fill the gap'.format(self._blocked_job_id, len(_to_fill)))
            # Filling the gap between cur_time and res_time
            (reserved_time, reserved_nodes) = self._reserved_slot
            filling_allocation = self.nonauto_allocator.allocating_method(_to_fill, cur_time, \
                                reserved_time=reserved_time,
                                reserved_nodes=[],
                                skip=True
                            )
            # Include the remaining jobs
            to_dispatch += filling_allocation        
        return to_dispatch, to_reject
    
    def _try_fifo_allocation(self, queued_jobs, cur_time):
        """
         Allocates as many jobs as possible using the FIFO approach. As soon as one allocation fails, all subsequent jobs fail too. 
         Then, the return tuple contains info about the allocated jobs (assigned nodes and such) and also the position of the blocked job.
        
         :param queued_jobs: List of job objects
         :param cur_time: current time
         
         :return job allocation, and position of the blocked job in the list
         
        """
       
        # Try to allocate jobs as in FIFO
        _allocated_jobs = self.nonauto_allocator.allocating_method(queued_jobs, cur_time, skip=False)
        
        # Check if there is a blocked job (a job without an allocation)
        blocked_idx = None    
        for i, (_, job_id, allocated_nodes) in enumerate(_allocated_jobs):
            if not allocated_nodes:
                self._blocked_job_id = job_id   
                blocked_idx = i
                break
        return _allocated_jobs, blocked_idx

    def _calculate_slot(self, cur_time, avl_resources, decided_allocations, prev_blocked, blocked_job, es_dict):   
        """
           Computes a reservation for the blocked job, by releasing incrementally the resources used by the running
           events and recently allocated jobs. The earliest slot in which blocked_job fits is chosen.
       
        :param avl_resources: Actual available resources
        :param decided_allocations: Allocated jobs on the current iteration.
        :param prev_blocked: Allocation corresponding to the previous blocked job which has been unblocked during this iteration
        :param blocked_jobs: Event to be fitted in the time slot
        :param es_dist: Job dictionary
       
        :return: a tuple of time of the slot and nodes
        """    
        
        current_allocations = self.resource_manager.current_allocations
        # Creates a list the jobs sorted by soonest ending time first
        future_endings = SortedListWithKey(key=lambda x:x[1])
                
        # Running jobs
        for job_id, resources in current_allocations.items():
            future_endings.add((job_id, es_dict[job_id].start_time + es_dict[job_id].expected_duration, resources))
        
        # Previous blocked job has been scheduled
        if prev_blocked:
            decided_allocations += prev_blocked
        
        # Current allocated job
        for (_, job_id, nodes) in decided_allocations:
            _dec_alloc = {}
            for node in nodes:
                if not(node in _dec_alloc):
                    _dec_alloc[node] = {k:v for k, v in es_dict[job_id].requested_resources.items()}
                else:
                    for res, v in es_dict[job_id].requested_resources.items():
                        _dec_alloc[node][res] += v
            future_endings.add((job_id, cur_time + es_dict[job_id].expected_duration, _dec_alloc))

        _required_alloc = blocked_job.requested_nodes
        _requested_resources = blocked_job.requested_resources
        _partial_alloc = {}

        # Calculate the partial allocation on the current system state
        for node, resources in avl_resources.items():
            new_alloc = min([resources[req] // _requested_resources[req] for req in _requested_resources])
            if new_alloc > 0:
                _partial_alloc[node] = new_alloc            
            
        # Calculate the partial allocation on the next future endings
        for (job_id, res_time, used_nodes) in future_endings:
            for node, used_resources in used_nodes.items():
                if not(node in avl_resources):
                    avl_resources[node] = {r:0 for r in _requested_resources}
                for r, v in used_resources.items():
                    avl_resources[node][r] += v
                
                cur_alloc = _partial_alloc.get(node, 0)
                new_alloc = min([avl_resources[node][req] // _requested_resources[req] for req in _requested_resources])
                _diff = new_alloc - cur_alloc
                if _diff > 0:
                    _partial_alloc[node] = _partial_alloc.get(node, 0) + _diff                                    
            
            # At this point the blocked job can be allocated
            if sum(_partial_alloc.values()) >= _required_alloc:
                ctimes = 0
                nodes = []
                for node, times in _partial_alloc.items():
                    ctimes += times
                    nodes.append(node)
                    if ctimes >= _required_alloc:
                        break
                return (res_time, nodes,)
        raise DispatcherError('Can\'t find the slot.... no end? :(')


### MODIFIED BY DAVIDE LEONE - START ###

import math
import numpy as np
import pandas as pd
import random

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.neighbors import KNeighborsClassifier

import random
import numpy as np
import pandas as pd
from sortedcontainers import SortedListWithKey
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from accasim.base.scheduler_class import SchedulerBase


class ShortestJobFirstExtended(SchedulerBase):
    """
    Unified Shortest Job First (SJF) scheduler with multiple runtime estimation modes.

    Modes:
    - 'base'          → Uses job.expected_duration (standard SJF)
    - 'oracle'        → Uses job.duration (true runtime)
    - 'error'         → Uses job.duration with injected estimation error
    - 'regressor'     → Predicts runtime using DecisionTree, RandomForest, or GradientBoosting
    - 'poly_regressor'→ Predicts runtime using Ridge regression with polynomial features
    - 'classifier'    → Classifies jobs into duration classes using k-NN
    - 'history'       → Uses average runtime per user from history

    Parameters (optional depending on mode):
    - probability, modification → For 'error' mode
    - training_data, additional_data, features, target_column, regressor → For regression modes
    - n_classes, advanced → For 'classifier' mode
    """

    def __init__(self,
                 _allocator,
                 mode='base',
                 training_data=None,
                 additional_data=None,
                 features=None,
                 target_column='Run Time',
                 regressor='DT',
                 n_classes=4,
                 advanced=True,
                 probability=0.0,
                 modification=0,
                 precomputed_data=None,
                 precomputed_column='pred_runtime_user',
                 job_id_column='job_id',
                 _seed=0,
                 **kwargs):

        super().__init__(_seed, _allocator, **kwargs)
        self.mode = mode.lower()
        self.prediction_cache = {}

        # === Parameters for all modes ===
        self.training_data = training_data
        self.additional_data = additional_data
        self.features = features
        self.target_column = target_column

        self.precomputed_data = precomputed_data
        self.precomputed_column = precomputed_column
        self.job_id_column = job_id_column
        
        # === Mode-specific initializations ===
        if self.mode == 'error':
            # Duration perturbation
            self._init_error_mode(probability, modification)
        elif self.mode in {'regressor', 'poly_regressor'}:
            # Regression-based estimation
            self._init_regressor_mode(regressor, _seed)
        elif self.mode == 'classifier':
            # Classification-based estimation
            self._init_classifier_mode(n_classes, advanced)
        elif self.mode == 'history':
            # User historical averages
            self._init_history_mode(training_data)
        elif self.mode == 'precomputed':
            self._init_precomputed_mode(n_classes, advanced)


        # Define sorting behavior (common to all modes)
        self.sorting_arguments = {'key': lambda job: self.get_predicted_runtime(job)}

    # ==============================================================
    # Mode Initializations
    # ==============================================================

    def _init_error_mode(self, probability, modification):
        if not (0 <= probability <= 1):
            raise ValueError(f"Probability must be between 0 and 1, got {probability}.")
        if modification < 0:
            raise ValueError(f"Modification must be >= 0, got {modification}.")
        self.probability = probability
        self.modification = modification

    def _init_regressor_mode(self, regressor_type, seed):
        if self.training_data is None or self.features is None:
            raise ValueError("training_data and features must be provided for regressor modes.")

        if self.mode == 'regressor':
            if regressor_type == "DT":
                self.regressor = DecisionTreeRegressor(random_state=seed)
            elif regressor_type == "RF":
                self.regressor = RandomForestRegressor(random_state=seed, n_estimators=100)
            elif regressor_type == "GB":
                self.regressor = GradientBoostingRegressor(random_state=seed, n_estimators=100, learning_rate=0.1)
            else:
                raise ValueError(f"Unsupported regressor type: {regressor_type}")
        elif self.mode == 'poly_regressor':
            self.regressor = self.RidgePolynomialRegressor(degree=2, alpha=0.01)

        self.regressor.fit(self.training_data[self.features], self.training_data[self.target_column])

    def _init_classifier_mode(self, n_classes, advanced):
        if self.training_data is None or self.features is None:
            raise ValueError("training_data and features must be provided for classifier mode.")
        self.n_classes = n_classes
        self.advanced = advanced

        # Map duration categories to numeric classes
        if n_classes == 4:
            mapping = {'Very-Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4}
        elif n_classes == 7:
            mapping = {'Very-Short': 1, 'Short': 2, 'Medium-Short': 3, 'Medium': 4,
                       'Medium-Long': 5, 'Long': 6, 'Very-Long': 7}
        else:
            raise ValueError("n_classes must be 4 or 7.")

        self.training_data['Duration'] = self.training_data['Duration'].map(mapping)
        self.classifier = KNeighborsClassifier(n_neighbors=n_classes)
        self.classifier.fit(self.training_data[self.features], self.training_data['Duration'])

    def _init_history_mode(self, training_data):
        if training_data is None:
            raise ValueError("training_data must be provided for history mode.")
        grouped = training_data.groupby('User ID')['Run Time'].mean()
        self.runtime_history = grouped.to_dict()
        self.user_job_counts = {}

    def _init_precomputed_mode(self, n_classes=None, advanced=False):
        if self.precomputed_data is None:
            raise ValueError("precomputed_data must be provided for precomputed mode.")
        if self.precomputed_column not in self.precomputed_data.columns:
            raise ValueError(f"Column '{self.precomputed_column}' not found in precomputed_data.")
        if self.job_id_column not in self.precomputed_data.columns:
            raise ValueError(f"Column '{self.job_id_column}' not found in precomputed_data.")

        self.n_classes = n_classes
        self.advanced = advanced

        # Label → class-id mapping (same as classifier)
        if n_classes == 4:
            self.label_to_class = {
                'Very-Short': 1,
                'Short': 2,
                'Medium': 3,
                'Long': 4
            }
        elif n_classes == 7:
            self.label_to_class = {
                'Very-Short': 1,
                'Short': 2,
                'Medium-Short': 3,
                'Medium': 4,
                'Medium-Long': 5,
                'Long': 6,
                'Very-Long': 7
            }
        elif n_classes is not None:
            raise ValueError("n_classes must be 4 or 7.")
            
        self.precomputed_predictions = (
            self.precomputed_data
            .assign(**{self.job_id_column: self.precomputed_data[self.job_id_column].astype(int)})
            .set_index(self.job_id_column)[self.precomputed_column]
            .to_dict()
        )

    # ==============================================================
    # Supporting Classes / Utilities
    # ==============================================================

    class RidgePolynomialRegressor:
        """Simple wrapper for Ridge regression with polynomial features."""
        def __init__(self, degree=2, alpha=0.01):
            self.degree = degree
            self.alpha = alpha
            self.poly = PolynomialFeatures(degree=degree, include_bias=True)
            self.scaler = StandardScaler()
            self.model = Ridge(alpha=self.alpha)

        def fit(self, X, y):
            X_poly = self.poly.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_poly)
            self.model.fit(X_scaled, y)

        def predict(self, X):
            X_poly = self.poly.transform(X)
            X_scaled = self.scaler.transform(X_poly)
            preds = self.model.predict(X_scaled)
            return np.clip(preds, 1, 86400).astype(int)

    # ==============================================================
    # Core logic — runtime prediction
    # ==============================================================

    def get_predicted_runtime(self, job):
        """Select runtime estimation strategy based on mode."""
        job_id = job.id

        # Cache predictions to speed up
        if job_id in self.prediction_cache:
            return self.prediction_cache[job_id]
        
        if self.mode == 'base':
            pred = job.expected_duration
        elif self.mode == 'oracle':
            pred = job.duration
        elif self.mode == 'error':
            pred = self._apply_duration_error(job)
        elif self.mode in {'regressor', 'poly_regressor'}:
            pred = int(self.regressor.predict(self._get_job_data(job))[0])
        elif self.mode == 'classifier':
            pred = self._predict_duration_class(job)
        elif self.mode == 'history':
            pred = self._predict_from_history(job)
        elif self.mode == 'precomputed':
            job_id = int(job.id)
            pred = self.precomputed_predictions.get(job_id)
        
            if pred is None:
                pred = job.expected_duration
        
            # --- runtime prediction ---
            if self.n_classes is None:
                pred = int(pred)
        
            # --- class-based prediction ---
            else:
                # Convert string labels if needed
                if isinstance(pred, str):
                    pred_class = self.label_to_class[pred]
                else:
                    pred_class = int(pred)
        
                if self.advanced:
                    pred = (pred_class, job.expected_duration)
                else:
                    pred = pred_class

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.prediction_cache[job_id] = pred
        return pred

    # --- Mode-specific helpers ---

    def _apply_duration_error(self, job):
        """Applies a random modification to job.duration based on probability and modification value."""
        if random.random() < self.probability:
            if self.modification == 0:
                return random.randint(1, 86400)
            elif self.modification > 1:
                return min(int(job.duration * self.modification), 86400)
            elif self.modification < 1:
                return max(int(job.duration * self.modification), 1)
        return job.duration

    def _get_job_data(self, job):
        """Builds a dataframe row for regression/classification models."""
        job_data = pd.DataFrame({
            'Job Number': [int(job.id)],
            'User ID': [int(job.user_id)],
            'Requested Number of Nodes': [int(job._requested_nodes)],
            'Requested Memory': [int(job._requested_resources['mem'])],
            'Requested Time': [int(job.expected_duration)]
        })

        if self.additional_data is not None:
            self.additional_data['Job Number'] = self.additional_data['Job Number'].astype(int)
            row = self.additional_data[self.additional_data['Job Number'] == int(job.id)]
            if not row.empty:
                job_data = job_data.merge(row)

        return job_data[self.features]

    def _predict_duration_class(self, job):
        """Predicts a duration class using the trained classifier."""
        job_data = self._get_job_data(job)
        pred_class = int(self.classifier.predict(job_data)[0])
        if self.advanced:
            return (pred_class, job.expected_duration)
        return pred_class

    def _predict_from_history(self, job):
        """Predicts runtime based on historical averages per user."""
        uid = job.user_id
        if uid in self.runtime_history:
            pred = self.runtime_history[uid]
        else:
            pred = job.expected_duration
        self._update_user_history(job)
        return pred

    def _update_user_history(self, job):
        """Incrementally updates user's average runtime."""
        uid = job.user_id
        dur = job.duration
        if uid not in self.runtime_history:
            self.runtime_history[uid] = dur
            self.user_job_counts[uid] = 1
        else:
            n = self.user_job_counts.get(uid, 1)
            old_avg = self.runtime_history[uid]
            new_avg = (old_avg * n + dur) / (n + 1)
            self.runtime_history[uid] = new_avg
            self.user_job_counts[uid] = n + 1

    # ==============================================================
    # Scheduling method
    # ==============================================================

    def scheduling_method(self, cur_time, jobs, es_dict):
        """Schedules jobs by ascending predicted runtime."""
        to_reject = []
        to_schedule = SortedListWithKey(jobs, **self.sorting_arguments)
        return to_schedule, to_reject

    def get_id(self):
        return f"{self.__class__.__name__}-{self.mode}-{self.allocator.get_id()}"



class EASYBackfillingExtended(SchedulerBase):
    """
    Unified EASY Backfilling scheduler with multiple runtime estimation modes.

    Modes:
    - 'base'          → Uses job.expected_duration (standard EBF)
    - 'oracle'        → Uses job.duration (true runtime)
    - 'regressor'     → Predicts runtime using DecisionTree, RandomForest, or GradientBoosting
    - 'poly_regressor'→ Predicts runtime using Ridge regression with polynomial features
    - 'classifier'    → Predicts duration using k-NN classifier mapping duration classes to representative times
    - 'history'       → Uses average runtime per user from historical data

    Parameters (optional depending on mode):
    - training_data, additional_data, features, target_column, regressor → For regression modes
    - n_classes, advanced → For 'classifier' mode
    """
    
    def __init__(self,
                 _allocator,
                 mode='base',
                 training_data=None,
                 additional_data=None,
                 features=None,
                 target_column='Run Time',
                 regressor='DT',
                 n_classes=4,
                 advanced=True,
                 precomputed_data=None,
                 precomputed_column='pred_runtime_user',
                 job_id_column='job_id',
                 _seed=0,
                 **kwargs):

        super().__init__(_seed, allocator=None, **kwargs)
        self.mode = mode.lower()

        # Allocator setup
        self._blocked_job_id = None
        self._reserved_slot = (None, [])
        self.nonauto_allocator = _allocator
        self.allocator_rm_set = False

        # Common data holders
        self.prediction_cache = {}
        self.training_data = training_data
        self.additional_data = additional_data
        self.features = features
        self.target_column = target_column

        self.precomputed_data = precomputed_data
        self.precomputed_column = precomputed_column
        self.job_id_column = job_id_column
        
        # Initialize mode-specific model
        if self.mode in {'regressor', 'poly_regressor'}:
            self._init_regressor_mode(regressor, seed)
        elif self.mode == 'classifier':
            self._init_classifier_mode(n_classes, advanced)
        elif self.mode == 'history':
            self._init_history_mode(training_data)
        elif self.mode == 'precomputed':
            self._init_precomputed_mode(n_classes, advanced)


    # ==============================================================
    # Mode initializations
    # ==============================================================

    def _init_regressor_mode(self, regressor_type, seed):
        if self.training_data is None or self.features is None:
            raise ValueError("training_data and features must be provided for regressor modes.")
        if self.mode == 'regressor':
            if regressor_type == "DT":
                self.regressor = DecisionTreeRegressor(random_state=seed)
            elif regressor_type == "RF":
                self.regressor = RandomForestRegressor(random_state=seed, n_estimators=100)
            elif regressor_type == "GB":
                self.regressor = GradientBoostingRegressor(random_state=seed, n_estimators=100, learning_rate=0.1)
            else:
                raise ValueError(f"Unsupported regressor type: {regressor_type}")
        else:  # poly_regressor
            self.regressor = self.RidgePolynomialRegressor(degree=2, alpha=0.01)
        self.regressor.fit(self.training_data[self.features], self.training_data[self.target_column])

    def _init_classifier_mode(self, n_classes, advanced):
        if self.training_data is None or self.features is None:
            raise ValueError("training_data and features must be provided for classifier mode.")
        self.n_classes = n_classes
        self.advanced = advanced
        
        # Map duration categories to representative times
        if n_classes == 4:
            mapping = {'Very-Short': 10, 'Short': 300, 'Medium': 7200, 'Long': 86400}
        elif n_classes == 7:
            mapping = {
                'Very-Short': 10, 'Short': 200, 'Medium-Short': 2000,
                'Medium': 6000, 'Medium-Long': 20000, 'Long': 50000, 'Very-Long': 86400
            }
        else:
            raise ValueError("n_classes must be 4 or 7.")

        self.training_data['Duration'] = self.training_data['Duration'].map(mapping)
        self.classifier = KNeighborsClassifier(n_neighbors=n_classes)
        self.classifier.fit(self.training_data[self.features], self.training_data['Duration'])

    def _init_history_mode(self, training_data):
        if training_data is None:
            raise ValueError("training_data must be provided for history mode.")
        grouped = training_data.groupby('User ID')['Run Time'].mean()
        self.runtime_history = grouped.to_dict()
        self.user_job_counts = training_data.groupby('User ID').size().to_dict()

    def _init_precomputed_mode(self, n_classes=None, advanced=False):
        if self.precomputed_data is None:
            raise ValueError("precomputed_data must be provided for precomputed mode.")
        if self.precomputed_column not in self.precomputed_data.columns:
            raise ValueError(f"Column '{self.precomputed_column}' not found in precomputed_data.")
        if self.job_id_column not in self.precomputed_data.columns:
            raise ValueError(f"Column '{self.job_id_column}' not found in precomputed_data.")
    
        self.n_classes = n_classes
        self.advanced = advanced
    
        # Label → class-id mapping (SAME as classifier)
        if n_classes == 4:
            self.label_to_class = {
                'Very-Short': 1,
                'Short': 2,
                'Medium': 3,
                'Long': 4
            }
            self.class_to_runtime = {
                1: 10,
                2: 300,
                3: 7200,
                4: 86400
            }
        elif n_classes == 7:
            self.label_to_class = {
                'Very-Short': 1,
                'Short': 2,
                'Medium-Short': 3,
                'Medium': 4,
                'Medium-Long': 5,
                'Long': 6,
                'Very-Long': 7
            }
            self.class_to_runtime = {
                1: 10,
                2: 200,
                3: 2000,
                4: 6000,
                5: 20000,
                6: 50000,
                7: 86400
            }
        elif n_classes is not None:
            raise ValueError("n_classes must be 4 or 7.")
    
        self.precomputed_predictions = (
            self.precomputed_data
            .assign(**{self.job_id_column: self.precomputed_data[self.job_id_column].astype(int)})
            .set_index(self.job_id_column)[self.precomputed_column]
            .to_dict()
        )

    # ==============================================================
    # Helper: polynomial regressor class
    # ==============================================================

    class RidgePolynomialRegressor:
        """Simple wrapper for Ridge regression with polynomial features."""
        def __init__(self, degree=2, alpha=0.01):
            self.degree = degree
            self.alpha = alpha
            self.poly = PolynomialFeatures(degree=degree, include_bias=True)
            self.scaler = StandardScaler()
            self.model = Ridge(alpha=self.alpha)

        def fit(self, X, y):
            X_poly = self.poly.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_poly)
            self.model.fit(X_scaled, y)

        def predict(self, X):
            X_poly = self.poly.transform(X)
            X_scaled = self.scaler.transform(X_poly)
            preds = self.model.predict(X_scaled)
            return np.clip(preds, 1, 86400).astype(int)

    # ==============================================================
    # Runtime prediction logic
    # ==============================================================

    def predict_runtime(self, job):
        """Selects runtime estimation strategy based on mode."""
        job_id = job.id
        if job_id in self.prediction_cache:
            return self.prediction_cache[job_id]

        if self.mode == 'base':
            pred = job.expected_duration
        elif self.mode == 'oracle':
            pred = job.duration
        elif self.mode in {'regressor', 'poly_regressor'}:
            pred = int(self.regressor.predict(self._get_job_data(job))[0])
        elif self.mode == 'classifier':
            pred = self._predict_duration_class(job)
        elif self.mode == 'history':
            pred = self._predict_from_history(job)
        elif self.mode == 'precomputed':
            job_id = int(job.id)
            pred = self.precomputed_predictions.get(job_id)
        
            if pred is None:
                pred = job.expected_duration
        
            # --- precomputed runtime ---
            if self.n_classes is None:
                pred = int(pred)
        
            # --- precomputed classifier ---
            else:
                # Convert string labels → numeric classes
                if isinstance(pred, str):
                    pred_class = self.label_to_class[pred]
                else:
                    pred_class = int(pred)
        
                base_runtime = self.class_to_runtime.get(
                    pred_class, job.expected_duration
                )
        
                if self.advanced:
                    pred = (base_runtime + job.expected_duration) / 2
                else:
                    pred = base_runtime

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.prediction_cache[job_id] = pred
        return pred

    def _predict_duration_class(self, job):
        job_data = self._get_job_data(job)
        pred = int(self.classifier.predict(job_data)[0])
        if self.advanced:
            pred = (pred + job_data['Requested Time'][0]) / 2
        return pred

    def _predict_from_history(self, job):
        uid = job.user_id
        pred = self.runtime_history.get(uid, job.expected_duration)
        self._update_user_history(job)
        return pred

    def _update_user_history(self, job):
        uid = job.user_id
        dur = job.duration
        if uid not in self.runtime_history:
            self.runtime_history[uid] = dur
            self.user_job_counts[uid] = 1
        else:
            n = self.user_job_counts.get(uid, 1)
            old_avg = self.runtime_history[uid]
            new_avg = (old_avg * n + dur) / (n + 1)
            self.runtime_history[uid] = new_avg
            self.user_job_counts[uid] = n + 1

    def _get_job_data(self, job):
        job_data = pd.DataFrame({
            'Job Number': [int(job.id)],
            'User ID': [int(job.user_id)],
            'Requested Number of Nodes': [int(job._requested_nodes)],
            'Requested Memory': [int(job._requested_resources['mem'])],
            'Requested Time': [int(job.expected_duration)]
        })

        if self.additional_data is not None:
            self.additional_data['Job Number'] = self.additional_data['Job Number'].astype(int)
            row = self.additional_data[self.additional_data['Job Number'] == int(job.id)]
            if not row.empty:
                job_data = job_data.merge(row)

        return job_data[self.features]

    # ==============================================================
    # Scheduling core (shared across all variants)
    # ==============================================================

    def scheduling_method(self, cur_time, queued_jobs, es_dict):
        """Implements the core EASY Backfilling scheduling loop."""
        if not self.allocator_rm_set:
            self.nonauto_allocator.set_resource_manager(self.resource_manager)
            self.allocator_rm_set = True

        avl_resources = self.resource_manager.current_availability
        self.nonauto_allocator.set_resources(avl_resources)

        to_dispatch, to_reject = [], []
        _to_fill, _prev_blocked = [], None
        _time_reached = False

        # 1. Handle reserved slot expiration
        if self._reserved_slot[0] and self._reserved_slot[0] <= cur_time:
            _time_reached = True
            blocked_job = queued_jobs[0]
            queued_jobs = queued_jobs[1:]
            allocation = self.nonauto_allocator.allocating_method(blocked_job, cur_time, skip=False)

            if allocation[-1]:
                self._blocked_job_id = None
                self._reserved_slot = (None, [])
                _prev_blocked = [allocation]

            to_dispatch += [allocation]

        # 2. Try FIFO allocation if no blocked job
        if self._blocked_job_id is None and queued_jobs:
            _allocated_jobs, blocked_idx = self._try_fifo_allocation(queued_jobs, cur_time)

            if blocked_idx is not None:
                if not self._reserved_slot[0]:
                    blocked_job = queued_jobs[blocked_idx]
                    self._reserved_slot = self._calculate_slot(
                        cur_time,
                        deepcopy(avl_resources),
                        _allocated_jobs[:blocked_idx],
                        _prev_blocked,
                        blocked_job,
                        es_dict
                    )
                to_dispatch += _allocated_jobs[:blocked_idx + 1]
                _to_fill = queued_jobs[blocked_idx + 1:]
            else:
                to_dispatch += _allocated_jobs
        else:
            if not _time_reached:
                to_dispatch += [(None, self._blocked_job_id, [])]
                _to_fill = queued_jobs[1:]
            else:
                _to_fill = queued_jobs

        # 3. Fill the gap with backfilling
        if _to_fill:
            reserved_time, reserved_nodes = self._reserved_slot
            filling_allocation = self.nonauto_allocator.allocating_method(
                _to_fill, cur_time, reserved_time=reserved_time, reserved_nodes=[], skip=True
            )
            to_dispatch += filling_allocation

        return to_dispatch, to_reject

    def _try_fifo_allocation(self, queued_jobs, cur_time):
        _allocated_jobs = self.nonauto_allocator.allocating_method(queued_jobs, cur_time, skip=False)
        blocked_idx = None
        for i, (_, job_id, allocated_nodes) in enumerate(_allocated_jobs):
            if not allocated_nodes:
                self._blocked_job_id = job_id
                blocked_idx = i
                break
        return _allocated_jobs, blocked_idx

    # ==============================================================
    # Reservation calculation
    # ==============================================================

    def _calculate_slot(self, cur_time, avl_resources, decided_allocations, prev_blocked, blocked_job, es_dict):
        """Computes the next available reservation for the blocked job."""
        current_allocations = self.resource_manager.current_allocations
        future_endings = SortedListWithKey(key=lambda x: x[1])

        # Running jobs
        for job_id, resources in current_allocations.items():
            runtime = self.predict_runtime(es_dict[job_id])
            future_endings.add((job_id, es_dict[job_id].start_time + runtime, resources))

        # Add newly allocated jobs
        if prev_blocked:
            decided_allocations += prev_blocked

        for (_, job_id, nodes) in decided_allocations:
            node_resources = {}
            for node in nodes:
                if node not in node_resources:
                    node_resources[node] = es_dict[job_id].requested_resources.copy()
                else:
                    for r, v in es_dict[job_id].requested_resources.items():
                        node_resources[node][r] += v
            runtime = self.predict_runtime(es_dict[job_id])
            future_endings.add((job_id, cur_time + runtime, node_resources))

        _required_alloc = blocked_job.requested_nodes
        _requested_resources = blocked_job.requested_resources
        _partial_alloc = {}

        # Initial allocation from available resources
        for node, resources in avl_resources.items():
            new_alloc = min(resources[req] // _requested_resources[req] for req in _requested_resources)
            if new_alloc > 0:
                _partial_alloc[node] = new_alloc

        # Incrementally release resources as jobs finish
        for (job_id, res_time, used_nodes) in future_endings:
            for node, used_resources in used_nodes.items():
                if node not in avl_resources:
                    avl_resources[node] = {r: 0 for r in _requested_resources}
                for r, v in used_resources.items():
                    avl_resources[node][r] += v

                cur_alloc = _partial_alloc.get(node, 0)
                new_alloc = min(avl_resources[node][req] // _requested_resources[req] for req in _requested_resources)
                if new_alloc > cur_alloc:
                    _partial_alloc[node] = new_alloc

            if sum(_partial_alloc.values()) >= _required_alloc:
                nodes = []
                ctimes = 0
                for node, times in _partial_alloc.items():
                    nodes.append(node)
                    ctimes += times
                    if ctimes >= _required_alloc:
                        break
                return res_time, nodes

        raise DispatcherError("Can't find the slot.... no end? :(")

    def get_id(self):
        return f"{self.__class__.__name__}-{self.mode}-{self.nonauto_allocator.name}"



class SmallestEnergyFirst(SimpleHeuristic):
    """
    Smallest Energy First (SEF) prioritizes jobs based on:
    
        Energy = (α * norm(expected_duration)) * (β * norm(requested_nodes))
    
    where normalization method is configurable:
    - 'minmax' 
    - 'zscore'
    - 'log' (default)
    - 'max'
    """

    name = 'SEF'

    def __init__(self, _allocator, _resource_manager=None, _seed=0,
                 min_duration=1, max_duration=86400,
                 min_nodes=1, max_nodes=980,
                 mean_duration=None, std_duration=None,
                 mean_nodes=None, std_nodes=None,
                 normalization='log',
                 alpha=1.0, beta=1.0,
                 **kwargs):

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

        self.mean_duration = mean_duration
        self.std_duration = std_duration
        self.mean_nodes = mean_nodes
        self.std_nodes = std_nodes

        self.normalization = normalization
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

        self.sorting_arguments = {
            'key': lambda job: self.compute_energy(job)
        }

        super().__init__(_seed, _allocator, self.name, self.sorting_arguments, **kwargs)

    def normalize(self, value, method, min_val=None, max_val=None, mean=None, std=None):
        if method == 'minmax':
            return (value - min_val) / (max_val - min_val + self.epsilon)
        elif method == 'zscore':
            return (value - mean) / (std + self.epsilon)
        elif method == 'log':
            return math.log(value + 1) / (math.log(max_val + 1) + self.epsilon)
        elif method == 'max':
            return value / (max_val + self.epsilon)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def compute_energy(self, job):
        norm_duration = self.normalize(
            value=job.expected_duration,
            method=self.normalization,
            min_val=self.min_duration,
            max_val=self.max_duration,
            mean=self.mean_duration,
            std=self.std_duration
        )
        norm_nodes = self.normalize(
            value=job.requested_nodes,
            method=self.normalization,
            min_val=self.min_nodes,
            max_val=self.max_nodes,
            mean=self.mean_nodes,
            std=self.std_nodes
        )
        return (self.alpha * norm_duration) * (self.beta * norm_nodes)



class SmallestEnergyFirstExtended(SchedulerBase):
    """
    Unified Smallest Energy First (SEF) scheduling policy with multiple runtime estimation modes.

    Modes:
    - 'base'          → Uses job.expected_duration
    - 'oracle'        → Uses job.duration
    - 'regressor'     → Predicts runtime via DecisionTree, RF, or GB regressor
    - 'poly_regressor'→ Predicts runtime via Ridge regression with polynomial features
    - 'classifier'    → Predicts duration class using k-NN
    - 'history'       → Uses average user runtime from history

    Energy formula:
        Energy = (α * norm(runtime)) * (β * norm(requested_nodes))

    Normalization methods:
    - 'minmax'
    - 'zscore'
    - 'log' (default)
    - 'max'
    - 'none' → no normalization
    """

    def __init__(self,
             _allocator,
             mode='base',
             training_data=None,
             additional_data=None,
             features=None,
             target_column='Run Time',
             regressor='DT',
             n_classes=4,
             advanced=True,
             normalization='log',
             alpha=1.0, beta=1.0,
             min_duration=1, max_duration=86400,
             min_nodes=1, max_nodes=980,
             mean_duration=None, std_duration=None,
             mean_nodes=None, std_nodes=None,
             precomputed_data=None,
             precomputed_column='pred_runtime_user',
             job_id_column='job_id',
             degree=2, reg_alpha=0.01,
             _seed=0,
             **kwargs):

        super().__init__(_seed, _allocator, **kwargs)
        self.mode = mode.lower()
        self.epsilon = 1e-6

        # Common normalization parameters
        self.normalization = normalization
        self.alpha = alpha
        self.beta = beta
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.mean_duration = mean_duration
        self.std_duration = std_duration
        self.mean_nodes = mean_nodes
        self.std_nodes = std_nodes

        # Mode-specific data
        self.training_data = training_data
        self.additional_data = additional_data
        self.features = features
        self.target_column = target_column
        self.prediction_cache = {}

        self.precomputed_data = precomputed_data
        self.precomputed_column = precomputed_column
        self.job_id_column = job_id_column

        # Initialize models depending on mode
        if self.mode in {'regressor', 'poly_regressor'}:
            self._init_regressor_mode(regressor, degree, reg_alpha, _seed)
        elif self.mode == 'classifier':
            self._init_classifier_mode(n_classes, advanced)
        elif self.mode == 'history':
            self._init_history_mode(training_data)
        elif self.mode == 'precomputed':
            self._init_precomputed_mode(n_classes, advanced)


        # Sorting key: energy-based ranking
        self.sorting_arguments = {'key': lambda job: self.compute_energy(job)}

    # ==============================================================
    # Mode initialization
    # ==============================================================

    def _init_regressor_mode(self, regressor_type, degree, alpha, seed):
        if self.training_data is None or self.features is None:
            raise ValueError("training_data and features must be provided for regressor modes.")
        if self.mode == 'regressor':
            if regressor_type == "DT":
                self.regressor = DecisionTreeRegressor(random_state=seed)
            elif regressor_type == "RF":
                self.regressor = RandomForestRegressor(random_state=seed, n_estimators=100)
            elif regressor_type == "GB":
                self.regressor = GradientBoostingRegressor(random_state=seed, n_estimators=100, learning_rate=0.1)
            else:
                raise ValueError(f"Unsupported regressor type: {regressor_type}")
        else:
            self.regressor = self.RidgePolynomialRegressor(degree=degree, alpha=alpha)

        self.regressor.fit(self.training_data[self.features], self.training_data[self.target_column])

    def _init_classifier_mode(self, n_classes, advanced):
        if self.training_data is None or self.features is None:
            raise ValueError("training_data and features must be provided for classifier mode.")
        self.n_classes = n_classes
        self.advanced = advanced
        self.classifier = KNeighborsClassifier(n_neighbors=n_classes)
        self.classifier.fit(self.training_data[self.features], self.training_data['Duration'])

    def _init_history_mode(self, training_data):
        if training_data is None:
            raise ValueError("training_data must be provided for history mode.")
        grouped = training_data.groupby('User ID')['Run Time'].mean()
        self.runtime_history = grouped.to_dict()
        self.user_job_counts = training_data['User ID'].value_counts().to_dict()

    def _init_precomputed_mode(self, n_classes=None, advanced=False):
        self.n_classes = n_classes
        self.advanced = advanced
    
        if n_classes == 4:
            self.label_to_class = {
                'Very-Short': 1,
                'Short': 2,
                'Medium': 3,
                'Long': 4
            }
            self.class_to_runtime = {
                1: 10,
                2: 300,
                3: 7200,
                4: 86400
            }
        elif n_classes == 7:
            self.label_to_class = {
                'Very-Short': 1,
                'Short': 2,
                'Medium-Short': 3,
                'Medium': 4,
                'Medium-Long': 5,
                'Long': 6,
                'Very-Long': 7
            }
            self.class_to_runtime = {
                1: 10,
                2: 200,
                3: 2000,
                4: 6000,
                5: 20000,
                6: 50000,
                7: 86400
            }
        elif n_classes is not None:
            raise ValueError("n_classes must be 4 or 7.")
    
        self.precomputed_predictions = (
            self.precomputed_data
            .assign(**{self.job_id_column: self.precomputed_data[self.job_id_column].astype(int)})
            .set_index(self.job_id_column)[self.precomputed_column]
            .to_dict()
        )


    # ==============================================================
    # Ridge Polynomial Regressor helper class
    # ==============================================================

    class RidgePolynomialRegressor:
        def __init__(self, degree=2, alpha=0.01):
            self.degree = degree
            self.alpha = alpha
            self.poly = PolynomialFeatures(degree=degree, include_bias=True)
            self.scaler = StandardScaler()
            self.model = Ridge(alpha=self.alpha)

        def fit(self, X, y):
            X_poly = self.poly.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_poly)
            self.model.fit(X_scaled, y)

        def predict(self, X):
            X_poly = self.poly.transform(X)
            X_scaled = self.scaler.transform(X_poly)
            preds = self.model.predict(X_scaled)
            return np.clip(preds, 1, 86400).astype(float)

    # ==============================================================
    # Normalization utilities
    # ==============================================================

    def normalize(self, value, method, min_val=None, max_val=None, mean=None, std=None):
        if method == 'none':
            return float(value)
    
        if method == 'minmax':
            return (value - min_val) / (max_val - min_val + self.epsilon)
        elif method == 'zscore':
            return (value - mean) / (std + self.epsilon)
        elif method == 'log':
            return math.log(value + 1) / (math.log(max_val + 1) + self.epsilon)
        elif method == 'max':
            return value / (max_val + self.epsilon)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    # ==============================================================
    # Runtime prediction logic
    # ==============================================================

    def get_predicted_runtime(self, job):
        job_id = job.id
        if job_id in self.prediction_cache:
            return self.prediction_cache[job_id]
    
        if self.mode == 'base':
            pred = job.expected_duration
        elif self.mode == 'oracle':
            pred = job.duration
        elif self.mode in {'regressor', 'poly_regressor'}:
            job_data = self._get_job_data(job)
            pred = float(self.regressor.predict(job_data)[0])
        elif self.mode == 'classifier':
            pred = self._predict_class_duration(job)
        elif self.mode == 'history':
            pred = self._predict_from_history(job)
        elif self.mode == 'precomputed':
            job_id = int(job.id)
            pred = self.precomputed_predictions.get(job_id)
        
            if pred is None:
                pred = job.expected_duration
        
            if self.n_classes is None:
                pred = int(pred)
            else:
                if isinstance(pred, str):
                    pred_class = self.label_to_class[pred]
                else:
                    pred_class = int(pred)
        
                base_runtime = self.class_to_runtime.get(
                    pred_class, job.expected_duration
                )
        
                if self.advanced:
                    pred = (base_runtime + job.expected_duration) / 2
                else:
                    pred = base_runtime

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
        self.prediction_cache[job_id] = pred
        return pred

    def _get_job_data(self, job):
        job_data = pd.DataFrame({
            'Job Number': [int(job.id)],
            'User ID': [int(job.user_id)],
            'Requested Number of Nodes': [int(job._requested_nodes)],
            'Requested Memory': [int(job._requested_resources['mem'])],
            'Requested Time': [int(job.expected_duration)]
        })
        if self.additional_data is not None:
            self.additional_data['Job Number'] = self.additional_data['Job Number'].astype(int)
            row = self.additional_data[self.additional_data['Job Number'] == int(job.id)]
            if not row.empty:
                job_data = job_data.merge(row)
        return job_data[self.features]

    def _predict_class_duration(self, job):
        job_data = self._get_job_data(job)
        pred_class = float(self.classifier.predict(job_data)[0])
        if self.advanced:
            return (pred_class + job.expected_duration) / 2
        return pred_class

    def _predict_from_history(self, job):
        uid = job.user_id
        pred = self.runtime_history.get(uid, job.expected_duration)
        self._update_user_history(job)
        return pred

    def _update_user_history(self, job):
        uid = job.user_id
        dur = job.duration
        if uid not in self.runtime_history:
            self.runtime_history[uid] = dur
            self.user_job_counts[uid] = 1
        else:
            n = self.user_job_counts.get(uid, 1)
            old_avg = self.runtime_history[uid]
            new_avg = (old_avg * n + dur) / (n + 1)
            self.runtime_history[uid] = new_avg
            self.user_job_counts[uid] = n + 1

    # ==============================================================
    # Core energy computation and scheduling
    # ==============================================================

    def compute_energy(self, job):
        predicted_runtime = self.get_predicted_runtime(job)
        requested_nodes = job.requested_nodes

        norm_dur = self.normalize(predicted_runtime, self.normalization,
                                  min_val=self.min_duration, max_val=self.max_duration,
                                  mean=self.mean_duration,
                                  std=self.std_duration)
        norm_nodes = self.normalize(requested_nodes, self.normalization,
                                    min_val=self.min_nodes, max_val=self.max_nodes,
                                    mean=self.mean_nodes, std=self.std_nodes)
        return (self.alpha * norm_dur) * (self.beta * norm_nodes)

    def scheduling_method(self, cur_time, jobs, es_dict):
        """Schedules jobs by ascending energy."""
        to_reject = []
        to_schedule = SortedListWithKey(jobs, **self.sorting_arguments)
        return to_schedule, to_reject

    def get_id(self):
        return f"{self.__class__.__name__}-{self.mode}-{self.allocator.get_id()}"


class prb_scheduler(SchedulerBase):
    """
    PRB type scheduler. Sorts the events depending on their expected and accumulated waiting time in the queue.
    
    In this scheduler, jobs can be skipped. If one fails, allocation is still tried on the following jobs.
    Sorting as name, sort funct parameters
    """
    name = 'PRB'
    def __init__(self, _allocator, _resource_manager=None, _seed=0, _ewt={'default': 1800}, **kwargs):
        SchedulerBase.__init__(self, _seed, allocator=_allocator, skip_jobs_on_allocation=True)
        self.ewt = _ewt

    def get_id(self):
        """
        Returns the full ID of the scheduler, including policy and allocator.

        :return: the scheduler's id.
        """
        return '-'.join([self.__class__.__name__, self.name, self.allocator.get_id()])

    def scheduling_method(self, cur_time, es, es_dict, _debug=False):
        """
        This function must map the queued events to available nodes at the current time.

        :param cur_time: current time
        :param es_dict: dictionary with full data of the events
        :param es: events to be scheduled
        :param _debug: Flag to debug

        :return a tuple of (time to schedule, event id, list of assigned nodes)  
        """
        avl_resources = self.resource_manager.current_availability
        # self.allocator.set_resources(avl_resources)

        # Sorted by more time in queue time, break ties with more simpliest requests
        sorted_es = self._sort_events(cur_time, es_dict, es)
        
        event_list = [es_dict[e.id] for e in sorted_es]
                
        return event_list, []

    def _get_ewt(self, queue_type):
        """
        Returns the expected waiting time for the selected queue.
        
        :param queue_type: the queue type
        :return: the expected waiting time
        """
        if queue_type in self.ewt:
            return self.ewt[queue_type]
        return self.ewt['default']

    def _sort_events(self, cur_time, events_dict, events):
        """
        Method which sorts the events depending on their waiting times in the queue.
        
        :param cur_time: the current time
        :param events_dict: the events dictionary
        :param events: the list of events to be scheduled
        :return: the sorted list of events
        """
        if len(events) <= 1:
            return events
        
        sort_helper = {
            e.id:
                {
                    'qtime': cur_time - e.queued_time + 1,
                    'ewt': self._get_ewt(e.queue),
                    'dur': e.expected_duration,
                    'req': sum([e.requested_nodes * val for attr, val in
                                e.requested_resources.items()])
                }
            for e in events
        }
        sort_helper['max_ewt'] = max([v['ewt'] for v in sort_helper.values()])
        
        return sorted(events, key=lambda e: (
        -(sort_helper['max_ewt'] * sort_helper[e.id]['qtime']) / sort_helper[e.id]['ewt'],
        sort_helper[e.id]['dur'] * sort_helper[e.id]['req']))


class PriorityRulesBasedExtended(prb_scheduler):
    """
    Extended PRB scheduler with support for precomputed runtimes and configurable tie-breaking.

    Features:
    1) Can load runtime ('dur') from a CSV (like SJF precomputed mode)
    2) Supports two tie-breaking strategies:
        - 'job_area'   → dur * req (default)
        - 'job_runtime'→ dur only

    Parameters:
    - precomputed_data=None → pandas DataFrame with predictions
    - precomputed_column='pred_runtime_user'
    - job_id_column='job_id'
    - tie_breaker='job_area'
    """

    name = 'PRB-EXT'

    def __init__(self,
                 _allocator,
                 _resource_manager=None,
                 _seed=0,
                 _ewt={'default': 1800},
                 precomputed_data=None,
                 precomputed_column='pred_runtime_user',
                 job_id_column='job_id',
                 tie_breaker='job_area',
                 **kwargs):

        super().__init__(_allocator,
                         _resource_manager=_resource_manager,
                         _seed=_seed,
                         _ewt=_ewt,
                         **kwargs)

        # --- Precomputed runtime parameters ---
        self.precomputed_data = precomputed_data
        self.precomputed_column = precomputed_column
        self.job_id_column = job_id_column

        # --- Tie breaker mode ---
        if tie_breaker not in {'job_area', 'job_runtime'}:
            raise ValueError("tie_breaker must be 'job_area' or 'job_runtime'")
        self.tie_breaker = tie_breaker

        # --- Cache for predictions ---
        self.prediction_cache = {}

        # --- Initialize precomputed runtimes if provided ---
        if self.precomputed_data is not None:
            if self.precomputed_column not in self.precomputed_data.columns:
                raise ValueError(f"Column '{self.precomputed_column}' not found in precomputed_data.")
            if self.job_id_column not in self.precomputed_data.columns:
                raise ValueError(f"Column '{self.job_id_column}' not found in precomputed_data.")

            self.precomputed_predictions = (
                self.precomputed_data
                .assign(**{
                    self.job_id_column: self.precomputed_data[self.job_id_column].astype(int)
                })
                .set_index(self.job_id_column)[self.precomputed_column]
                .to_dict()
            )
        else:
            self.precomputed_predictions = None

    # ==============================================================
    # Runtime retrieval (like SJF precomputed mode)
    # ==============================================================

    def _get_runtime(self, job):
        """
        Returns runtime to be used in scheduling:
        - from precomputed CSV if available
        - otherwise expected_duration
        """
        job_id = int(job.id)

        if job_id in self.prediction_cache:
            return self.prediction_cache[job_id]

        if self.precomputed_predictions is not None:
            dur = self.precomputed_predictions.get(job_id)
            if dur is None:
                dur = job.expected_duration
            else:
                dur = int(dur)
        else:
            dur = job.expected_duration

        self.prediction_cache[job_id] = dur
        return dur

    # ==============================================================
    # Override sorting logic
    # ==============================================================

    def _sort_events(self, cur_time, events_dict, events):
        """
        Sort events using:
        1) PRB priority rule (same as base class)
        2) Custom tie-breaker:
            - job_area   → dur * req
            - job_runtime→ dur
        """
        if len(events) <= 1:
            return events

        sort_helper = {}

        for e in events:
            dur = self._get_runtime(e)

            sort_helper[e.id] = {
                'qtime': cur_time - e.queued_time + 1,
                'ewt': self._get_ewt(e.queue),
                'dur': dur,
                'req': sum([
                    e.requested_nodes * val
                    for attr, val in e.requested_resources.items()
                ])
            }

        sort_helper['max_ewt'] = max([v['ewt'] for v in sort_helper.values()])

        # --- Define tie breaker ---
        if self.tie_breaker == 'job_area':
            tie_fn = lambda e: sort_helper[e.id]['dur'] * sort_helper[e.id]['req']
        else:  # 'job_runtime'
            tie_fn = lambda e: sort_helper[e.id]['dur']

        return sorted(events, key=lambda e: (
            -(sort_helper['max_ewt'] * sort_helper[e.id]['qtime']) / sort_helper[e.id]['ewt'],
            tie_fn(e)
        ))

    # ==============================================================
    # ID
    # ==============================================================

    def get_id(self):
        base = super().get_id()
        mode = "precomputed" if self.precomputed_predictions is not None else "default"
        return f"{base}-{mode}-{self.tie_breaker}"



# ==============================================================
# ALIAS DEFINITION FOR BACKWARD COMPATIBILITY
# ==============================================================

# Originally, I implemented a variant for each behaviour, but then I merged them into three unified classes.
# These aliases ensure that all legacy scheduler variants continue to work exactly as before, while internally redirecting to the new unified classes:
#  - ShortestJobFirstExtended
#  - EASYBackfillingExtended
#  - SmallestEnergyFirstExtended
# This part is just to ensure that old experiments still work; new users can ignore it.

ShortestJobFirstWithOracle = lambda *args, **kwargs: ShortestJobFirstExtended(*args, mode="oracle", **kwargs)
ShortestJobFirstWithError = lambda *args, **kwargs: ShortestJobFirstExtended(*args, mode="error", **kwargs)
ShortestJobFirstWithRegressor = lambda *args, **kwargs: ShortestJobFirstExtended(*args, mode="regressor", **kwargs)
ShortestJobFirstWithPolynomialRegressor = lambda *args, **kwargs: ShortestJobFirstExtended(*args, mode="poly_regressor", **kwargs)
ShortestJobFirstWithClassifier = lambda *args, **kwargs: ShortestJobFirstExtended(*args, mode="classifier", **kwargs)
ShortestJobFirstWithHistory = lambda *args, **kwargs: ShortestJobFirstExtended(*args, mode="history", **kwargs)

EASYBackfillingWithOracle = lambda *args, **kwargs: EASYBackfillingExtended(*args, mode="oracle", **kwargs)
EASYBackfillingWithRegressor = lambda *args, **kwargs: EASYBackfillingExtended(*args, mode="regressor", **kwargs)
EASYBackfillingWithPolynomialRegressor = lambda *args, **kwargs: EASYBackfillingExtended(*args, mode="poly_regressor", **kwargs)
EASYBackfillingWithClassifier = lambda *args, **kwargs: EASYBackfillingExtended(*args, mode="classifier", **kwargs)
EASYBackfillingWithHistory = lambda *args, **kwargs: EASYBackfillingExtended(*args, mode="history", **kwargs)

SmallestEnergyFirstWithOracle = lambda *args, **kwargs: SmallestEnergyFirstExtended(*args, mode="oracle", **kwargs)
SmallestEnergyFirstWithRegressor = lambda *args, **kwargs: SmallestEnergyFirstExtended(*args, mode="regressor", **kwargs)
SmallestEnergyFirstWithPolynomialRegressor = lambda *args, **kwargs: SmallestEnergyFirstExtended(*args, mode="poly_regressor", **kwargs)
SmallestEnergyFirstWithClassifier = lambda *args, **kwargs: SmallestEnergyFirstExtended(*args, mode="classifier", **kwargs)
SmallestEnergyFirstWithHistory = lambda *args, **kwargs: SmallestEnergyFirstExtended(*args, mode="history", **kwargs)
        
        
### MODIFIED BY DAVIDE LEONE - END ###
