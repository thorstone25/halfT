function z = pagemtimes(x, y)
[x, y] = deal(halfT(x), halfT(y)); % enforce halfT type

if x.gtype || y.gtype % send to GPU
    [x, y] = deal(gpuArray(x), gpuArray(y));
else
    error('halfT/pagemtimes unsupported on the CPU (performance).');
end
    zc = ~(isreal(x) && isreal(y));
    zi = halfT(0);
    zr = gpu_pagemtimes_helper(real(x), real(y));
    if zc
        zr = zr - gpu_pagemtimes_helper(imag(x), imag(y)); end
    if ~isreal(y)
        zi = gpu_pagemtimes_helper(real(x), imag(y)); end
    if ~isreal(x)
        zi = zi + gpu_pagemtimes_helper(imag(x), real(y)); end
    if ~zc, z = zr; else, z = complex(zr, zi); end
end

% TODO: convert all inputs to half
% TODO: avoid empties, but broadcast proper dimensions?

