import type {
    initialize as Iinitialize,
    CARROT_Postprocessing as ICARROT_Postprocessing,
    TreeringPostprocessingResult,
    CellsPostprocessingResult,
    CombinedPostprocessingResult,
    CellInfo,
    PairedPaths,
    PathPair,
    Path,
    Point,
    ImageSize,
} from "./carrot-wasm-postprocessing.d.ts"

import { 
    is_array_of_type,
    is_number_array,
    is_object,
    has_number_property,
    has_property_of_type,
} from "./dep.ts"



// for better readability
type fn_pointer = number;
type pointer = number;


type CARROT_Postprocessing_WASM = {

    _postprocess_combined_wasm: (
        cellmap_filesize:                      number,
        cellmap_read_file_callback_p:          fn_pointer,
        cellmap_read_file_callback_handle:     number,
        treeringmap_filesize:                  number,
        treeringmap_read_file_callback_p:      fn_pointer,
        treeringmap_read_file_callback_handle: number,
        workshape_width:      number,
        workshape_height:     number,
        og_shape_width:       number,
        og_shape_height:      number,
        // outputs
        cellmap_workshape_png:           pointer,
        cellmap_workshape_png_size:      pointer,
        instancemap_workshape_png:       pointer,
        instancemap_workshape_png_size:  pointer,
        treeringmap_workshape_png:       pointer,
        treeringmap_workshape_png_size:  pointer,
        treeringmap_og_shape_png:        pointer,
        treeringmap_og_shape_png_size:   pointer,
        ring_points_xy_json:             pointer,
        ring_points_xy_json_size:        pointer,
        ringmap_workshape_png_pp:        pointer,
        ringmap_workshape_png_size_p:    pointer,
        cell_info_json_size:             pointer,
        cell_info_json:                  pointer,

        returncode: pointer,
    ) => number,

    _free_output: (p:pointer) => void,


    _malloc: (nbytes:number) => pointer,
    _free:   (ptr:pointer) => void,

    // deno-lint-ignore no-explicit-any
    addFunction: ( fn: (...args: any[]) => unknown, argtypes:string ) => number,
    
    HEAPU8: {
        set: (src:Uint8Array, dst:pointer) => void,
        slice: (start:number, end:number) => Uint8Array<ArrayBuffer>,
        [i:number]: number,
    }
    HEAP32: {
        [i:number]: number,
    },
    HEAP64: {
        [i:number]: bigint,
    },

    Asyncify: {
        // deno-lint-ignore no-explicit-any
        handleAsync: (fn:(...args:any[]) => Promise<unknown>) => unknown,
        currData: number|null,
    },
}


function wait(ms: number): Promise<unknown> {
    return new Promise((resolve) => {
        setTimeout(() => resolve(0), ms)
    })
}



export class CARROT_Postprocessing implements ICARROT_Postprocessing {
    constructor(private wasm:CARROT_Postprocessing_WASM){
        this.#read_file_callback_ptr = 
            wasm.addFunction(this.#read_file_callback, 'iiijj');
    }


    async postprocess_combined(
        cellmap:     File|null,
        treeringmap: File|null,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<CombinedPostprocessingResult | CellsPostprocessingResult | TreeringPostprocessingResult | Error> {

        let cells_handle:number = 0;
        let rings_handle:number = 0;
        if(cellmap){
            cells_handle = this.#handle_counter++;
            this.#read_file_callback_table[cells_handle] = cellmap;
        }
        if(treeringmap){
            rings_handle = this.#handle_counter++;
            this.#read_file_callback_table[rings_handle] = treeringmap;
        }


        const rc_ptr:pointer = this.#malloc(4, /*fill=*/255)

        const cellmap_workshape_png_pp:pointer         = this.#malloc(8);
        const cellmap_workshape_png_size_p:pointer     = this.#malloc(8);
        const instancemap_workshape_png_pp:pointer     = this.#malloc(8);
        const instancemap_workshape_png_size_p:pointer = this.#malloc(8);
        const treeringmap_workshape_png_pp:pointer     = this.#malloc(8);
        const treeringmap_workshape_png_size_p:pointer = this.#malloc(8);
        const treeringmap_og_shape_png_pp:pointer      = this.#malloc(8);
        const treeringmap_og_shape_png_size_p:pointer  = this.#malloc(8);
        const ring_points_xy_json_pp:pointer           = this.#malloc(8);
        const ring_points_xy_json_size_p:pointer       = this.#malloc(8);
        const ringmap_workshape_png_pp:pointer         = this.#malloc(8);
        const ringmap_workshape_png_size_p:pointer     = this.#malloc(8);
        const cell_info_json_pp:pointer                = this.#malloc(8);
        const cell_info_json_size_p:pointer            = this.#malloc(8);

        let cellmap_workshape_png_p:pointer|undefined     = undefined;
        let instancemap_workshape_png_p:pointer|undefined = undefined;
        let treeringmap_workshape_png_p:pointer|undefined = undefined;
        let treeringmap_og_shape_png_p:pointer|undefined  = undefined;
        let ring_points_xy_json_p:pointer|undefined       = undefined;
        let ringmap_workshape_png_p:pointer|undefined     = undefined;
        let cell_info_json_p:pointer|undefined            = undefined;


        try {
            let rc:number = await this.wasm._postprocess_combined_wasm(
                cellmap?.size ?? 0, 
                this.#read_file_callback_ptr, 
                cells_handle, 
                treeringmap?.size ?? 0, 
                this.#read_file_callback_ptr, 
                rings_handle, 

                work_size.width, 
                work_size.height,
                og_size.width,
                og_size.height,

                cellmap_workshape_png_pp,
                cellmap_workshape_png_size_p,
                instancemap_workshape_png_pp,
                instancemap_workshape_png_size_p,
                treeringmap_workshape_png_pp,
                treeringmap_workshape_png_size_p,
                treeringmap_og_shape_png_pp,
                treeringmap_og_shape_png_size_p,
                ring_points_xy_json_pp,
                ring_points_xy_json_size_p,
                ringmap_workshape_png_pp,
                ringmap_workshape_png_size_p,
                cell_info_json_pp,
                cell_info_json_size_p,

                rc_ptr
            )
            // NOTE: the wasm function above returns before the execution is 
            // finished because of async issues, so currently polling until done
            while(this.wasm.Asyncify.currData != null)
                await wait(1);
    
            rc = (rc == 0)? this.wasm.HEAP32[rc_ptr >> 2]! : rc;
            if(rc != 0)
                return new Error(`WASM error code = ${rc}`)

            let cellmap_workshape_png_u8:Uint8Array<ArrayBuffer>|null = null;
            let instancemap_workshape_png_u8:Uint8Array<ArrayBuffer>|null = null;
            if(cellmap) {
                cellmap_workshape_png_p = 
                    this.wasm.HEAP32[cellmap_workshape_png_pp >> 2]!;
                const cellmap_workshape_png_size:number = 
                    Number(this.wasm.HEAP64[cellmap_workshape_png_size_p >> 3]);
                cellmap_workshape_png_u8 = this.wasm.HEAPU8.slice(
                    cellmap_workshape_png_p, 
                    cellmap_workshape_png_p + cellmap_workshape_png_size
                )

                instancemap_workshape_png_p = 
                    this.wasm.HEAP32[instancemap_workshape_png_pp >> 2]!;
                const instancemap_workshape_png_size:number = 
                    Number(this.wasm.HEAP64[instancemap_workshape_png_size_p >> 3]);
                instancemap_workshape_png_u8 = this.wasm.HEAPU8.slice(
                    instancemap_workshape_png_p, 
                    instancemap_workshape_png_p + instancemap_workshape_png_size
                )
            }


            let treeringmap_workshape_png_u8:Uint8Array<ArrayBuffer>|null = null
            let treeringmap_og_shape_png_u8:Uint8Array<ArrayBuffer>|null = null
            let paired_paths:PairedPaths|null = null;
            if(treeringmap) {
                treeringmap_workshape_png_p = 
                    this.wasm.HEAP32[treeringmap_workshape_png_pp >> 2]!;
                const treeringmap_workshape_png_size:number = 
                    Number(this.wasm.HEAP64[treeringmap_workshape_png_size_p >> 3]);
                treeringmap_workshape_png_u8 = this.wasm.HEAPU8.slice(
                    treeringmap_workshape_png_p, 
                    treeringmap_workshape_png_p + treeringmap_workshape_png_size
                )

                treeringmap_og_shape_png_p = 
                    this.wasm.HEAP32[treeringmap_og_shape_png_pp >> 2]!;
                const treeringmap_og_shape_png_size:number = 
                    Number(this.wasm.HEAP64[treeringmap_og_shape_png_size_p >> 3]);
                treeringmap_og_shape_png_u8 = this.wasm.HEAPU8.slice(
                    treeringmap_og_shape_png_p, 
                    treeringmap_og_shape_png_p + treeringmap_og_shape_png_size
                )


                ring_points_xy_json_p = this.wasm.HEAP32[ring_points_xy_json_pp >> 2]!;
                const ring_points_xy_json_size:number = 
                    Number(this.wasm.HEAP64[ring_points_xy_json_size_p >> 3]);
                const ring_points_xy_json_u8:Uint8Array = this.wasm.HEAPU8.slice(
                    ring_points_xy_json_p, 
                    ring_points_xy_json_p + ring_points_xy_json_size
                )
                const obj:unknown = 
                    JSON.parse(new TextDecoder().decode(ring_points_xy_json_u8));
                paired_paths = validate_paired_paths(obj)
                if(paired_paths == null)
                    return new Error('WASM-JS communication inconcistencies')
            }


            let ringmap_workshape_png_u8:Uint8Array<ArrayBuffer>|null = null;
            let cell_info:CellInfo[]|null = null;
            if(cellmap && treeringmap) {
                ringmap_workshape_png_p = 
                    this.wasm.HEAP32[ringmap_workshape_png_pp >> 2]!;
                const ringmap_workshape_png_size:number = 
                    Number(this.wasm.HEAP64[ringmap_workshape_png_size_p >> 3]);
                ringmap_workshape_png_u8 = this.wasm.HEAPU8.slice(
                    ringmap_workshape_png_p, 
                    ringmap_workshape_png_p + ringmap_workshape_png_size
                )

                cell_info_json_p = this.wasm.HEAP32[cell_info_json_pp >> 2]!;
                const cell_info_json_size_p_json_size:number = 
                    Number(this.wasm.HEAP64[cell_info_json_size_p >> 3]);
                const cell_info_json_u8:Uint8Array = this.wasm.HEAPU8.slice(
                    cell_info_json_p, 
                    cell_info_json_p + cell_info_json_size_p_json_size
                )
                const obj:unknown = 
                    JSON.parse(new TextDecoder().decode(cell_info_json_u8));
                cell_info = validate_cell_info_array(obj)
                if(cell_info == null)
                    return new Error('WASM-JS communication inconcistencies')
            }

            if(cellmap_workshape_png_u8 
            && instancemap_workshape_png_u8 
            && treeringmap_workshape_png_u8 
            && paired_paths 
            && cell_info
            && ringmap_workshape_png_u8)
                return {
                    cellmap_workshape_png: 
                        new File([cellmap_workshape_png_u8], 'cellmap.png'),
                    instancemap_workshape_png: 
                        new File([instancemap_workshape_png_u8], 'instancemap.png'),
                    
                    treeringmap_workshape_png:
                        new File([treeringmap_workshape_png_u8], 'treeringmap.png'),
                    ring_points_xy: paired_paths,

                    ringmap_workshape_png:
                        new File([ringmap_workshape_png_u8], 'ringmap.png'),
                    cell_info: cell_info,

                    _type: "combined",
                };
            else if(cellmap_workshape_png_u8 && instancemap_workshape_png_u8)
                return {
                    cellmap_workshape_png: 
                        new File([cellmap_workshape_png_u8], 'cellmap.png'),
                    instancemap_workshape_png: 
                        new File([instancemap_workshape_png_u8], 'instancemap.png'),
                    
                    _type: "cells"
                }
            else if(treeringmap_workshape_png_u8 
                 && treeringmap_og_shape_png_u8 
                 && paired_paths)
                return {
                    treeringmap_workshape_png:
                        new File([treeringmap_workshape_png_u8], 'treeringmap.png'),
                    treeringmap_og_shape_png:
                        new File([treeringmap_og_shape_png_u8], 'treeringmap.png'),
                    ring_points_xy: paired_paths,

                    _type: "treerings"
                }
            else
                return new Error('Unexpected error')

        } catch (e) {
            console.error('Unexpected error:', e)
            return e as Error;
        } finally {
            this.wasm._free(rc_ptr);
            this.wasm._free(cellmap_workshape_png_pp);
            this.wasm._free(cellmap_workshape_png_size_p);
            this.wasm._free(instancemap_workshape_png_pp);
            this.wasm._free(instancemap_workshape_png_size_p);
            this.wasm._free(treeringmap_workshape_png_pp);
            this.wasm._free(treeringmap_workshape_png_size_p);
            this.wasm._free(treeringmap_og_shape_png_pp);
            this.wasm._free(treeringmap_og_shape_png_size_p);
            this.wasm._free(ring_points_xy_json_pp);
            this.wasm._free(ring_points_xy_json_size_p);
            this.wasm._free(ringmap_workshape_png_pp);
            this.wasm._free(ringmap_workshape_png_size_p);
            this.wasm._free(cell_info_json_pp);
            this.wasm._free(cell_info_json_size_p);

            if(cells_handle != 0)
                delete this.#read_file_callback_table[cells_handle];
            if(rings_handle != 0)
                delete this.#read_file_callback_table[rings_handle];

            if(cellmap_workshape_png_p != undefined) 
                this.wasm._free_output(cellmap_workshape_png_pp);
            if(instancemap_workshape_png_p != undefined) 
                this.wasm._free_output(instancemap_workshape_png_pp);
            if(treeringmap_workshape_png_p != undefined) 
                this.wasm._free_output(treeringmap_workshape_png_pp);
            if(treeringmap_og_shape_png_p != undefined)
                this.wasm._free_output(treeringmap_og_shape_png_pp);
            if(ringmap_workshape_png_p != undefined) 
                this.wasm._free_output(ringmap_workshape_png_pp);
            if(ring_points_xy_json_p != undefined) 
                this.wasm._free_output(ring_points_xy_json_pp);
            if(cell_info_json_p != undefined) 
                this.wasm._free_output(cell_info_json_pp);
        }
    }





    #handle_counter = 1;

    #read_file_callback_ptr:pointer;
    #read_file_callback_table: Record<number, File> = {};

    /** Called by WASM to read a required portion of a file */
    #read_file_callback = (
        handle: number,
        dstbuf: pointer,
        start:  bigint,
        size:   bigint,
    ): unknown  => {
        return this.wasm.Asyncify.handleAsync( async () => {
            const file:File|undefined = 
                this.#read_file_callback_table[handle];
            if(!file)
                return -1;
            
            const slice_u8:Uint8Array = new Uint8Array(
                await file.slice(Number(start), Number(start+size)).arrayBuffer()
            )
            this.wasm.HEAPU8.set(slice_u8, dstbuf);
            return 0;
        })
    }

    #malloc(nbytes:number, fill?:number): pointer {
        const p:pointer = this.wasm._malloc(nbytes);
        this.wasm.HEAPU8.set(new Uint8Array(nbytes).fill(fill ?? 0), p)
        return p;
    }
}


function validate_point(x:unknown): Point|null {
    if(is_number_array(x) && x.length == 2){
        return x as Point;
    }
    else return null;
}

function validate_path(x:unknown): Path|null {
    if(is_array_of_type(x, validate_point)){
        return x;
    }
    else return null;
}

function validate_pathpair(x:unknown): PathPair|null {
    if(is_array_of_type(x, validate_path) && x.length == 2){
        return x as PathPair;
    }
    else return null;
}

function validate_paired_paths(x:unknown):PairedPaths|null {
    if(is_array_of_type(x, validate_pathpair)){
        return x;
    }
    else return null;
}


function validate_box(x:unknown): [number,number,number,number]|null {
    if(is_number_array(x) && x.length == 4)
            return x as [number,number,number,number];
    else return  null;
}

function validate_cell_info_object(x:unknown): CellInfo|null {
    if(is_object(x)
    && has_number_property(x, "id")
    && has_property_of_type(x, "box_xy", validate_box)
    && has_number_property(x, "year_index")
    && has_number_property(x, "area")
    && has_number_property(x, "position_within")
    ){
        return x;
    }
    else return null;
}

function validate_cell_info_array(x:unknown): CellInfo[]|null {
    if(is_array_of_type(x, validate_cell_info_object)){
        return x;
    }
    else return null;
}




export const initialize:typeof Iinitialize = async () => {
    const wasm:CARROT_Postprocessing_WASM = await (
        await import('./build-wasm/carrot_postprocessing_wasm.js')
    // deno-lint-ignore no-explicit-any
    ).default() as any;

    return new CARROT_Postprocessing(wasm);
}


