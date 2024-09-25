/* Copyright (c) 2023, MariaDB

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1335  USA */


/**
  @file

  @brief
  This file defines all vector functions
*/

#include <x86intrin.h>
#include <my_global.h>
#include "item.h"
#include "item_vectorfunc.h"


key_map Item_func_vec_distance::part_of_sortkey() const
{
  key_map map(0);
  if (Item_field *item= get_field_arg())
  {
    Field *f= item->field;
    for (uint i= f->table->s->keys; i < f->table->s->total_keys; i++)
      if (f->table->s->key_info[i].algorithm == HA_KEY_ALG_MHNSW &&
          f->key_start.is_set(i))
        map.set_bit(i);
  }
  return map;
}

double Item_func_vec_distance::val_real()
{
  String *r1= args[0]->val_str();
  String *r2= args[1]->val_str();
  null_value= !r1 || !r2 || r1->length() != r2->length() ||
              r1->length() % sizeof(float);
  if (null_value)
    return 0;
  float *v1= (float*)r1->ptr();
  float *v2= (float*)r2->ptr();
  return euclidean_vec_distance(v1, v2, (r1->length()) / sizeof(float));
}

double euclidean_vec_distance(float *v1, float *v2, size_t v_len)
{
    float *p1 = v1;
    float *p2 = v2;
    __m128d d = _mm_setzero_pd();

    size_t i;
    // process 4 elems per cycle
    for(i = 0; i < v_len - 3; i += 4)
    {
        const __m128 a = _mm_loadu_ps(p1 + i);
        const __m128 b = _mm_loadu_ps(p2 + i);

        const __m128d c = _mm_cvtps_pd(_mm_sub_ps(a, b)); // c = a - b
        const __m128d dist = _mm_mul_pd(c, c); // dist = c * c

        d = _mm_add_pd(d, dist); // d += dist
    }

    // process vectors' tail (3 elems at max)
    for(; i < v_len; ++i)
    {
        const __m128 a = _mm_load_ss(p1 + i);
        const __m128 b = _mm_load_ss(p2 + i);

        const __m128d c = _mm_cvtps_pd(_mm_sub_ps(a, b));
        const __m128d dist = _mm_mul_pd(c, c);

        d = _mm_add_pd(d, dist);
    }

    d = _mm_hadd_pd(d, d);
    d = _mm_hadd_pd(d, d);
    return _mm_cvtsd_f64(_mm_unpackhi_pd(d, d));
}
