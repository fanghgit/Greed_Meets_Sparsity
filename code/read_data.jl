function read_libsvm(filename)
    f = open(filename, "r")

    row_ft = zeros(Int, 0)
    col_ft = zeros(Int, 0)
    val_ft = zeros(Float64, 0)

    y = zeros(Int, 0)
    
    
    cc = 1
    for line in eachline(f)
        #println(cc)
        tmp = split(line, " ")
        if length(tmp) <= 1
            println("weird at line: ", cc)
            continue
        end

        if tmp[1] != ""
            label = parse( Int, tmp[1] )
            push!( y, label )
        else
            error("not label at line: ", cc)
        end

        
        if tmp[2] == ""
            continue
        end

        for i = 2:length(tmp)
            if tmp[i] == ""
                continue 
            end
            
            idx, val = split(tmp[i], ":")
            idx = parse(Int, idx)
            val = parse(Float64, val)
            push!(col_ft, idx )
            push!(val_ft, val)
            push!(row_ft, cc)
        end
        cc += 1
    end
    close(f)

    ft_mat = sparse( row_ft, col_ft, val_ft )

    return ft_mat, y

end
