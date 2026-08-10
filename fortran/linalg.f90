
module linalg
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none (type, external)
    private
    public :: matmatmul_c

contains

    ! ------------- CORE FUNCTIONS ------------- !

    function vecmatmul(a, b) result(res)
        real, intent(in) :: a(:), b(:, :)
        real :: res(size(b, 1))
        integer :: i, j

        do i = 1, size(b, 1)
            res(i) = 0.0
            do j = 1, size(a)
                res(i) = res(i) + a(j) * b(i, j)
            end do
        end do
    end function vecmatmul

    function dot(a, b) result(res)
        real, intent(in) :: a(:), b(:)
        real :: res
        integer :: i

        res = 0.0
        do i = 1, size(a)
            res = res + a(i) * b(i)
        end do
    end function dot

    function outer(a, b) result(res)
        real, intent(in) :: a(:), b(:)
        real :: res(size(a), size(b))
        integer :: i, j

        do i = 1, size(a)
            do j = 1, size(b)
                res(i, j) = a(i) * b(j)
            end do
        end do
    end function outer

    ! ------------- OPTIMIZED SUBROUTINE WRAPPERS ------------- !

    subroutine matmatmul_c(a, b, res, n, m, p) bind(C, name="matmatmul_c")
        integer(c_int), intent(in), value :: n, m, p
        real(c_double), intent(in) :: a(n, m), b(m, p)
        real(c_double), intent(out) :: res(n, p)

        res = matmul(a, b)

    end subroutine matmatmul_c

    subroutine matvecmul_c(a, b, res, n, m) bind(C, name="matvecmul_c")
        use, intrinsic :: iso_c_binding, only: c_double, c_int
        implicit none (type, external)

        integer(c_int), intent(in), value :: n, m
        real(c_double), intent(in) :: a(n, m), b(m)
        real(c_double), intent(out) :: res(n)

        ! Because 'a' is incoming from a C-contiguous Python array,
        ! it will look transposed to Fortran. To perform Matrix * Vector (A * b),
        ! we must mathematically multiply Vector * Matrix (b * A) inside Fortran.
        res = matmul(b, a)

    end subroutine matvecmul_c

end module linalg
