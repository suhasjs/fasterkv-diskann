﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System.Runtime.CompilerServices;

namespace FASTER.core
{
    public unsafe partial class FasterKV<Key, Value> : FasterBase, IFasterKV<Key, Value>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal Status InternalContainsKeyInMemory<Input, Output, Context, FasterSession>(
            ref Key key, FasterSession fasterSession, out long logicalAddress, long fromAddress = -1)
            where FasterSession : IFasterSession<Key, Value, Input, Output, Context>
        {
            OperationStackContext<Key, Value> stackCtx = new(comparer.GetHashCode64(ref key));

            if (fasterSession.Ctx.phase != Phase.REST)
                HeavyEnter(stackCtx.hei.hash, fasterSession.Ctx, fasterSession);

            if (FindTag(ref stackCtx.hei))
            {
                stackCtx.SetRecordSourceToHashEntry(hlog);

                if (UseReadCache)
                    SkipReadCache(ref stackCtx, out _);

                if (fromAddress < hlog.HeadAddress)
                    fromAddress = hlog.HeadAddress;

                if (TryFindRecordInMainLog(ref key, ref stackCtx, fromAddress) && !stackCtx.recSrc.GetInfo().Tombstone)
                {
                    logicalAddress = stackCtx.recSrc.LogicalAddress;
                    return new(StatusCode.Found);
                }
                logicalAddress = 0;
                return new(StatusCode.NotFound);
            }

            // no tag found
            logicalAddress = 0;
            return new(StatusCode.NotFound);
        }
    }
}
