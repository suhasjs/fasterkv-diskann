﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace FASTER.core
{
    public unsafe partial class FasterKV<Key, Value> : FasterBase, IFasterKV<Key, Value>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void SynchronizeEpoch<Input, Output, Context, FasterSession>(
            FasterExecutionContext<Input, Output, Context> sessionCtx,
            ref PendingContext<Input, Output, Context> pendingContext,
            FasterSession fasterSession)
            where FasterSession : IFasterSession<Key, Value, Input, Output, Context>
        {
            var version = sessionCtx.version;
            Debug.Assert(sessionCtx.version == version);
            Debug.Assert(sessionCtx.phase == Phase.PREPARE);
            InternalRefresh<Input, Output, Context, FasterSession>(fasterSession);
            Debug.Assert(sessionCtx.version > version);

            pendingContext.version = sessionCtx.version;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void HeavyEnter<Input, Output, Context, FasterSession>(long hash, FasterExecutionContext<Input, Output, Context> ctx, FasterSession session)
            where FasterSession : IFasterSession<Key, Value, Input, Output, Context>
        {
            if (ctx.phase == Phase.PREPARE_GROW)
            {
                // We spin-wait as a simplification
                // Could instead do a "heavy operation" here
                while (systemState.Phase != Phase.IN_PROGRESS_GROW)
                    Thread.SpinWait(100);
                InternalRefresh<Input, Output, Context, FasterSession>(session);
            }
            if (ctx.phase == Phase.IN_PROGRESS_GROW)
            {
                SplitBuckets(hash);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        void SpinWaitUntilClosed(long address)
        {
            // Unlike HeadAddress, ClosedUntilAddress is a high-water mark; a record that is == to ClosedUntilAddress has *not* been closed yet.
            while (address >= this.hlog.ClosedUntilAddress)
            {
                Debug.Assert(address < hlog.HeadAddress, "expected address < hlog.HeadAddress");
                epoch.ProtectAndDrain();
                Thread.Yield();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        void SpinWaitUntilRecordIsClosed(long logicalAddress, AllocatorBase<Key, Value> log)
        {
            Debug.Assert(logicalAddress < log.HeadAddress, "SpinWaitUntilRecordIsClosed should not be called for addresses above HeadAddress");

            // The caller checks for logicalAddress < HeadAddress, so we must ProtectAndDrain at least once, even if logicalAddress >= ClosedUntilAddress.
            // Otherwise, let's say we have two OnPagesClosed operations in the epoch drainlist:
            //      old epoch: old HeadAddress -> [ClosedUntilAddress is somewhere in here] -> intermediate HeadAddress
            //      middle epoch: we hold this
            //      newer epoch: intermediate HeadAddress -> current HeadAddress
            // If we find the record above ClosedUntilAddress and return immediately, the caller will retry -- but we haven't given the second
            // OnPagesClosed a chance to run and we're holding an epoch below it, so the caller will again see logicalAddress < HeadAddress... forever.
            // However, we don't want the caller to check for logicalAddress < ClosedUntilAddress instead of HeadAddress; this allows races where
            // lock are split between the record and the LockTable:
            //      lock -> eviction to lock table -> unlock -> ClosedUntilAddress updated
            // So the caller has to check for logicalAddress < HeadAddress and we have to run this loop at least once.
            while (true)
            {
                epoch.ProtectAndDrain();
                Thread.Yield();

                // Unlike HeadAddress, ClosedUntilAddress is a high-water mark; a record that is == to ClosedUntilAddress has *not* been closed yet.
                if (logicalAddress < log.ClosedUntilAddress)
                    break;

                // Note: We cannot jump out here if the Lock Table contains the key, because it may be an older version of the record.
            }
        }
    }
}
