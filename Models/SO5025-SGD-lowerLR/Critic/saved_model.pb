°°
Ż£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.0-dev202007112v1.12.1-36285-g51be86b23b8ė

critic_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_namecritic_model/dense_2/kernel

/critic_model/dense_2/kernel/Read/ReadVariableOpReadVariableOpcritic_model/dense_2/kernel*
_output_shapes

:2*
dtype0

critic_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2**
shared_namecritic_model/dense_2/bias

-critic_model/dense_2/bias/Read/ReadVariableOpReadVariableOpcritic_model/dense_2/bias*
_output_shapes
:2*
dtype0

critic_model/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_namecritic_model/dense_3/kernel

/critic_model/dense_3/kernel/Read/ReadVariableOpReadVariableOpcritic_model/dense_3/kernel*
_output_shapes

:2*
dtype0

critic_model/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecritic_model/dense_3/bias

-critic_model/dense_3/bias/Read/ReadVariableOpReadVariableOpcritic_model/dense_3/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
“
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ļ
valueåBā BŪ


dense1
	value
opt
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

	kernel

bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
6
iter
	decay
learning_rate
momentum

	0

1
2
3
 

	0

1
2
3
­
trainable_variables
layer_regularization_losses

layers
non_trainable_variables
regularization_losses
	variables
layer_metrics
metrics
 
YW
VARIABLE_VALUEcritic_model/dense_2/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEcritic_model/dense_2/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
­
trainable_variables
layer_regularization_losses

layers
 non_trainable_variables
regularization_losses
	variables
!layer_metrics
"metrics
XV
VARIABLE_VALUEcritic_model/dense_3/kernel'value/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcritic_model/dense_3/bias%value/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
#layer_regularization_losses

$layers
%non_trainable_variables
regularization_losses
	variables
&layer_metrics
'metrics
A?
VARIABLE_VALUESGD/iter#opt/iter/.ATTRIBUTES/VARIABLE_VALUE
CA
VARIABLE_VALUE	SGD/decay$opt/decay/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUESGD/learning_rate,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUESGD/momentum'opt/momentum/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
Æ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1critic_model/dense_2/kernelcritic_model/dense_2/biascritic_model/dense_3/kernelcritic_model/dense_3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_2884400
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ź
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/critic_model/dense_2/kernel/Read/ReadVariableOp-critic_model/dense_2/bias/Read/ReadVariableOp/critic_model/dense_3/kernel/Read/ReadVariableOp-critic_model/dense_3/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOpConst*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_2884526
Å
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecritic_model/dense_2/kernelcritic_model/dense_2/biascritic_model/dense_3/kernelcritic_model/dense_3/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentum*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_2884560Ōæ
ą
Æ
D__inference_dense_2_layer_call_and_return_conditional_losses_2884431

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisŃ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’22	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’:::S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ģ
~
)__inference_dense_3_layer_call_fn_2884479

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallų
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_28843542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
Į

I__inference_critic_model_layer_call_and_return_conditional_losses_2884371
input_1
dense_2_2884319
dense_2_2884321
dense_3_2884365
dense_3_2884367
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinput_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

ExpandDims£
dense_2/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0dense_2_2884319dense_2_2884321*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_28843082!
dense_2/StatefulPartitionedCallø
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_2884365dense_3_2884367*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_28843542!
dense_3/StatefulPartitionedCallÄ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
¤

 __inference__traced_save_2884526
file_prefix:
6savev2_critic_model_dense_2_kernel_read_readvariableop8
4savev2_critic_model_dense_2_bias_read_readvariableop:
6savev2_critic_model_dense_3_kernel_read_readvariableop8
4savev2_critic_model_dense_3_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6b8008695116497caa2df7dff187f479/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameå
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*÷
valueķBź	B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB#opt/iter/.ATTRIBUTES/VARIABLE_VALUEB$opt/decay/.ATTRIBUTES/VARIABLE_VALUEB,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB'opt/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesĄ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_critic_model_dense_2_kernel_read_readvariableop4savev2_critic_model_dense_2_bias_read_readvariableop6savev2_critic_model_dense_3_kernel_read_readvariableop4savev2_critic_model_dense_3_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2		2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes.
,: :2:2:2:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
ą
Æ
D__inference_dense_2_layer_call_and_return_conditional_losses_2884308

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisŃ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’22	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’:::S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Æ
D__inference_dense_3_layer_call_and_return_conditional_losses_2884470

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisŃ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::S O
+
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
¾%
Ė
#__inference__traced_restore_2884560
file_prefix0
,assignvariableop_critic_model_dense_2_kernel0
,assignvariableop_1_critic_model_dense_2_bias2
.assignvariableop_2_critic_model_dense_3_kernel0
,assignvariableop_3_critic_model_dense_3_bias
assignvariableop_4_sgd_iter 
assignvariableop_5_sgd_decay(
$assignvariableop_6_sgd_learning_rate#
assignvariableop_7_sgd_momentum

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7ė
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*÷
valueķBź	B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB'value/kernel/.ATTRIBUTES/VARIABLE_VALUEB%value/bias/.ATTRIBUTES/VARIABLE_VALUEB#opt/iter/.ATTRIBUTES/VARIABLE_VALUEB$opt/decay/.ATTRIBUTES/VARIABLE_VALUEB,opt/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB'opt/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesŲ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity«
AssignVariableOpAssignVariableOp,assignvariableop_critic_model_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_critic_model_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_critic_model_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_critic_model_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4 
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5”
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6©
AssignVariableOp_6AssignVariableOp$assignvariableop_6_sgd_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¤
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


%__inference_signature_wrapper_2884400
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallń
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_28842712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1

Æ
D__inference_dense_3_layer_call_and_return_conditional_losses_2884354

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisŃ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’2:::S O
+
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ĮW
®
"__inference__wrapped_model_2884271
input_1:
6critic_model_dense_2_tensordot_readvariableop_resource8
4critic_model_dense_2_biasadd_readvariableop_resource:
6critic_model_dense_3_tensordot_readvariableop_resource8
4critic_model_dense_3_biasadd_readvariableop_resource
identity|
critic_model/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
critic_model/ExpandDims/dim„
critic_model/ExpandDims
ExpandDimsinput_1$critic_model/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
critic_model/ExpandDimsÕ
-critic_model/dense_2/Tensordot/ReadVariableOpReadVariableOp6critic_model_dense_2_tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02/
-critic_model/dense_2/Tensordot/ReadVariableOp
#critic_model/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#critic_model/dense_2/Tensordot/axes
#critic_model/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#critic_model/dense_2/Tensordot/free
$critic_model/dense_2/Tensordot/ShapeShape critic_model/ExpandDims:output:0*
T0*
_output_shapes
:2&
$critic_model/dense_2/Tensordot/Shape
,critic_model/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,critic_model/dense_2/Tensordot/GatherV2/axisŗ
'critic_model/dense_2/Tensordot/GatherV2GatherV2-critic_model/dense_2/Tensordot/Shape:output:0,critic_model/dense_2/Tensordot/free:output:05critic_model/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'critic_model/dense_2/Tensordot/GatherV2¢
.critic_model/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.critic_model/dense_2/Tensordot/GatherV2_1/axisĄ
)critic_model/dense_2/Tensordot/GatherV2_1GatherV2-critic_model/dense_2/Tensordot/Shape:output:0,critic_model/dense_2/Tensordot/axes:output:07critic_model/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)critic_model/dense_2/Tensordot/GatherV2_1
$critic_model/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$critic_model/dense_2/Tensordot/ConstŌ
#critic_model/dense_2/Tensordot/ProdProd0critic_model/dense_2/Tensordot/GatherV2:output:0-critic_model/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#critic_model/dense_2/Tensordot/Prod
&critic_model/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&critic_model/dense_2/Tensordot/Const_1Ü
%critic_model/dense_2/Tensordot/Prod_1Prod2critic_model/dense_2/Tensordot/GatherV2_1:output:0/critic_model/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%critic_model/dense_2/Tensordot/Prod_1
*critic_model/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*critic_model/dense_2/Tensordot/concat/axis
%critic_model/dense_2/Tensordot/concatConcatV2,critic_model/dense_2/Tensordot/free:output:0,critic_model/dense_2/Tensordot/axes:output:03critic_model/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%critic_model/dense_2/Tensordot/concatą
$critic_model/dense_2/Tensordot/stackPack,critic_model/dense_2/Tensordot/Prod:output:0.critic_model/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$critic_model/dense_2/Tensordot/stacké
(critic_model/dense_2/Tensordot/transpose	Transpose critic_model/ExpandDims:output:0.critic_model/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’2*
(critic_model/dense_2/Tensordot/transposeó
&critic_model/dense_2/Tensordot/ReshapeReshape,critic_model/dense_2/Tensordot/transpose:y:0-critic_model/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2(
&critic_model/dense_2/Tensordot/Reshapeņ
%critic_model/dense_2/Tensordot/MatMulMatMul/critic_model/dense_2/Tensordot/Reshape:output:05critic_model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’22'
%critic_model/dense_2/Tensordot/MatMul
&critic_model/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22(
&critic_model/dense_2/Tensordot/Const_2
,critic_model/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,critic_model/dense_2/Tensordot/concat_1/axis¦
'critic_model/dense_2/Tensordot/concat_1ConcatV20critic_model/dense_2/Tensordot/GatherV2:output:0/critic_model/dense_2/Tensordot/Const_2:output:05critic_model/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'critic_model/dense_2/Tensordot/concat_1ä
critic_model/dense_2/TensordotReshape/critic_model/dense_2/Tensordot/MatMul:product:00critic_model/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’22 
critic_model/dense_2/TensordotĖ
+critic_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp4critic_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+critic_model/dense_2/BiasAdd/ReadVariableOpŪ
critic_model/dense_2/BiasAddBiasAdd'critic_model/dense_2/Tensordot:output:03critic_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’22
critic_model/dense_2/BiasAdd
critic_model/dense_2/ReluRelu%critic_model/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
critic_model/dense_2/ReluÕ
-critic_model/dense_3/Tensordot/ReadVariableOpReadVariableOp6critic_model_dense_3_tensordot_readvariableop_resource*
_output_shapes

:2*
dtype02/
-critic_model/dense_3/Tensordot/ReadVariableOp
#critic_model/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#critic_model/dense_3/Tensordot/axes
#critic_model/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#critic_model/dense_3/Tensordot/free£
$critic_model/dense_3/Tensordot/ShapeShape'critic_model/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$critic_model/dense_3/Tensordot/Shape
,critic_model/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,critic_model/dense_3/Tensordot/GatherV2/axisŗ
'critic_model/dense_3/Tensordot/GatherV2GatherV2-critic_model/dense_3/Tensordot/Shape:output:0,critic_model/dense_3/Tensordot/free:output:05critic_model/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'critic_model/dense_3/Tensordot/GatherV2¢
.critic_model/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.critic_model/dense_3/Tensordot/GatherV2_1/axisĄ
)critic_model/dense_3/Tensordot/GatherV2_1GatherV2-critic_model/dense_3/Tensordot/Shape:output:0,critic_model/dense_3/Tensordot/axes:output:07critic_model/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)critic_model/dense_3/Tensordot/GatherV2_1
$critic_model/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$critic_model/dense_3/Tensordot/ConstŌ
#critic_model/dense_3/Tensordot/ProdProd0critic_model/dense_3/Tensordot/GatherV2:output:0-critic_model/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#critic_model/dense_3/Tensordot/Prod
&critic_model/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&critic_model/dense_3/Tensordot/Const_1Ü
%critic_model/dense_3/Tensordot/Prod_1Prod2critic_model/dense_3/Tensordot/GatherV2_1:output:0/critic_model/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%critic_model/dense_3/Tensordot/Prod_1
*critic_model/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*critic_model/dense_3/Tensordot/concat/axis
%critic_model/dense_3/Tensordot/concatConcatV2,critic_model/dense_3/Tensordot/free:output:0,critic_model/dense_3/Tensordot/axes:output:03critic_model/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%critic_model/dense_3/Tensordot/concatą
$critic_model/dense_3/Tensordot/stackPack,critic_model/dense_3/Tensordot/Prod:output:0.critic_model/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$critic_model/dense_3/Tensordot/stackš
(critic_model/dense_3/Tensordot/transpose	Transpose'critic_model/dense_2/Relu:activations:0.critic_model/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’22*
(critic_model/dense_3/Tensordot/transposeó
&critic_model/dense_3/Tensordot/ReshapeReshape,critic_model/dense_3/Tensordot/transpose:y:0-critic_model/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2(
&critic_model/dense_3/Tensordot/Reshapeņ
%critic_model/dense_3/Tensordot/MatMulMatMul/critic_model/dense_3/Tensordot/Reshape:output:05critic_model/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%critic_model/dense_3/Tensordot/MatMul
&critic_model/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&critic_model/dense_3/Tensordot/Const_2
,critic_model/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,critic_model/dense_3/Tensordot/concat_1/axis¦
'critic_model/dense_3/Tensordot/concat_1ConcatV20critic_model/dense_3/Tensordot/GatherV2:output:0/critic_model/dense_3/Tensordot/Const_2:output:05critic_model/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'critic_model/dense_3/Tensordot/concat_1ä
critic_model/dense_3/TensordotReshape/critic_model/dense_3/Tensordot/MatMul:product:00critic_model/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2 
critic_model/dense_3/TensordotĖ
+critic_model/dense_3/BiasAdd/ReadVariableOpReadVariableOp4critic_model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+critic_model/dense_3/BiasAdd/ReadVariableOpŪ
critic_model/dense_3/BiasAddBiasAdd'critic_model/dense_3/Tensordot:output:03critic_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’2
critic_model/dense_3/BiasAdd}
IdentityIdentity%critic_model/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::::P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ģ
~
)__inference_dense_2_layer_call_fn_2884440

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallų
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_28843082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’22

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
²
¢
.__inference_critic_model_layer_call_fn_2884385
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_critic_model_layer_call_and_return_conditional_losses_28843712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1"ŹL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Æ
serving_default
;
input_10
serving_default_input_1:0’’’’’’’’’@
output_14
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:Ų=
ą

dense1
	value
opt
trainable_variables
regularization_losses
	variables
	keras_api

signatures
(_default_save_signature
*)&call_and_return_all_conditional_losses
*__call__"
_tf_keras_modelź{"class_name": "CriticModel", "name": "critic_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CriticModel"}}
ļ

	kernel

bias
trainable_variables
regularization_losses
	variables
	keras_api
*+&call_and_return_all_conditional_losses
,__call__"Ź
_tf_keras_layer°{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [41, 1, 1]}}
ņ

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*-&call_and_return_all_conditional_losses
.__call__"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [41, 1, 50]}}
I
iter
	decay
learning_rate
momentum"
	optimizer
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
Ź
trainable_variables
layer_regularization_losses

layers
non_trainable_variables
regularization_losses
	variables
layer_metrics
metrics
*__call__
(_default_save_signature
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
,
/serving_default"
signature_map
-:+22critic_model/dense_2/kernel
':%22critic_model/dense_2/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
trainable_variables
layer_regularization_losses

layers
 non_trainable_variables
regularization_losses
	variables
!layer_metrics
"metrics
,__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
-:+22critic_model/dense_3/kernel
':%2critic_model/dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
trainable_variables
#layer_regularization_losses

$layers
%non_trainable_variables
regularization_losses
	variables
&layer_metrics
'metrics
.__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ą2Ż
"__inference__wrapped_model_2884271¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
2
I__inference_critic_model_layer_call_and_return_conditional_losses_2884371Ę
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
ü2ł
.__inference_critic_model_layer_call_fn_2884385Ę
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
ī2ė
D__inference_dense_2_layer_call_and_return_conditional_losses_2884431¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_2_layer_call_fn_2884440¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_dense_3_layer_call_and_return_conditional_losses_2884470¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_3_layer_call_fn_2884479¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
4B2
%__inference_signature_wrapper_2884400input_1
"__inference__wrapped_model_2884271q	
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "7Ŗ4
2
output_1&#
output_1’’’’’’’’’°
I__inference_critic_model_layer_call_and_return_conditional_losses_2884371c	
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’
 
.__inference_critic_model_layer_call_fn_2884385V	
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "’’’’’’’’’¬
D__inference_dense_2_layer_call_and_return_conditional_losses_2884431d	
3¢0
)¢&
$!
inputs’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’2
 
)__inference_dense_2_layer_call_fn_2884440W	
3¢0
)¢&
$!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’2¬
D__inference_dense_3_layer_call_and_return_conditional_losses_2884470d3¢0
)¢&
$!
inputs’’’’’’’’’2
Ŗ ")¢&

0’’’’’’’’’
 
)__inference_dense_3_layer_call_fn_2884479W3¢0
)¢&
$!
inputs’’’’’’’’’2
Ŗ "’’’’’’’’’„
%__inference_signature_wrapper_2884400|	
;¢8
¢ 
1Ŗ.
,
input_1!
input_1’’’’’’’’’"7Ŗ4
2
output_1&#
output_1’’’’’’’’’